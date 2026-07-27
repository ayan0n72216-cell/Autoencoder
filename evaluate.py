import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估卷积自编码器的重建质量和理论存储大小"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="每批测试图片数量（默认：128）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="CIFAR-10 数据保存位置（默认：data）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("批大小必须是大于 0 的整数。")

    # 自动选择设备：有可用显卡时使用图形处理器，否则使用中央处理器。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用设备：图形处理器（GPU），{torch.cuda.get_device_name(0)}")
    else:
        print("使用设备：中央处理器（CPU）")

    checkpoint_path = Path("checkpoints") / "conv_autoencoder.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型参数文件：{checkpoint_path}。"
            "请先运行 python train.py 完成训练。"
        )

    # 与 train.py 和 reconstruct.py 相同，只使用 ToTensor()。
    # ToTensor() 会把 CIFAR-10 的像素值从整数转换到 [0, 1] 范围。
    transform = transforms.ToTensor()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ConvAutoencoder().to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)

    # eval() 会让模型在量化时执行 torch.round()，而不是添加训练噪声。
    model.eval()

    squared_error_sum = 0.0
    image_value_count = 0
    tested_image_count = 0
    image_height = 0
    image_width = 0

    # 字典的键是量化后的整数，值是该整数出现的总次数。
    quantized_value_counts: dict[int, int] = {}
    quantized_element_count = 0

    # 测试时不需要计算梯度，可以减少显存或内存占用。
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )

            reconstructed_images, latent, quantized_latent = model(images)

            # 当前模型的输入使用 [0, 1]，解码器末尾也使用 Sigmoid。
            # clamp 是一道额外保护，确保质量指标只在有效像素范围内计算。
            reconstructed_images = reconstructed_images.clamp(0.0, 1.0)

            # 先累计每个 RGB 数值的平方误差总和，最后再除以所有数值的
            # 总数。这样最后一个较小批次不会和完整批次得到相同权重。
            squared_error = (reconstructed_images - images) ** 2
            squared_error_sum += squared_error.sum().item()
            image_value_count += images.numel()
            tested_image_count += images.size(0)
            image_height = images.size(2)
            image_width = images.size(3)

            # 每批分别统计后只保存计数，不保存所有量化张量，避免占用显存。
            unique_values, counts = torch.unique(
                quantized_latent,
                return_counts=True,
            )
            unique_values = unique_values.cpu()
            counts = counts.cpu()

            for value, count in zip(unique_values.tolist(), counts.tolist()):
                # model.eval() 下经过 torch.round() 的值一定是整数。
                integer_value = int(value)
                quantized_value_counts[integer_value] = (
                    quantized_value_counts.get(integer_value, 0) + int(count)
                )

            quantized_element_count += quantized_latent.numel()

    if tested_image_count == 0 or image_value_count == 0:
        raise RuntimeError("测试集为空，无法计算评估结果。")
    if quantized_element_count == 0:
        raise RuntimeError("量化后的中间数据为空，无法估计理论大小。")

    # 整个测试集所有图片、通道和像素位置上的均方误差。
    average_reconstruction_error = squared_error_sum / image_value_count

    # 输入和重建图都在 [0, 1]，所以峰值信噪比的最大像素值取 1。
    if average_reconstruction_error == 0.0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10(1.0 / average_reconstruction_error)

    # 根据全测试集整数的经验概率计算香农熵。
    average_bits_per_latent_element = 0.0
    for count in quantized_value_counts.values():
        probability = count / quantized_element_count
        average_bits_per_latent_element -= probability * math.log2(probability)

    estimated_total_bits = (
        average_bits_per_latent_element * quantized_element_count
    )
    estimated_total_bytes = estimated_total_bits / 8.0

    # 这里按图像压缩中常用的“每个空间像素位数”统计，因此分母只用
    # 图片数量 × 高度 × 宽度，不再乘 RGB 颜色通道数。一个空间像素包含
    # 三个颜色分量，而量化后的 latent 表示的是整张彩色图片。
    original_spatial_pixel_count = (
        tested_image_count * image_height * image_width
    )
    average_bits_per_original_pixel = (
        estimated_total_bits / original_spatial_pixel_count
    )

    print("\n量化后中间数据的整数频数：")
    for integer_value in sorted(quantized_value_counts):
        count = quantized_value_counts[integer_value]
        print(f"  整数 {integer_value}：{count:,} 次")

    print("\n整个 CIFAR-10 测试集的评估结果：")
    print(f"测试图片数量：{tested_image_count:,}")
    print(
        "平均重建误差（均方误差，MSE）："
        f"{average_reconstruction_error:.8f}"
    )
    print(f"峰值信噪比（PSNR）：{psnr:.4f} 分贝（dB）")
    print(f"量化后中间数据的元素总数：{quantized_element_count:,}")
    print(
        "量化后出现的不同整数种类数："
        f"{len(quantized_value_counts):,}"
    )
    print(
        "平均每个中间数据元素的理论位数："
        f"{average_bits_per_latent_element:.6f} 位"
    )
    print(f"估计总位数：{estimated_total_bits:,.2f} 位")
    print(f"估计总字节数：{estimated_total_bytes:,.2f} 字节")
    print(
        "平均每个原图像素所需的理论位数："
        f"{average_bits_per_original_pixel:.6f} 位/像素"
    )

    print(
        "\n说明：以上大小是根据量化整数出现概率计算出的理论估计大小。"
    )
    print(
        "本脚本没有执行真正的熵编码，也没有生成压缩文件，"
        "因此这些数值不能当作实际文件大小。"
    )


if __name__ == "__main__":
    main()
