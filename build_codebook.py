import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from huffman_encoding import build_huffman_codebook
from model import ConvAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用整个 CIFAR-10 训练集建立公共霍夫曼编码表"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="每批训练图片数量（默认：128）",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用设备：图形处理器（GPU），{torch.cuda.get_device_name(0)}")
    else:
        print("使用设备：中央处理器（CPU）")

    checkpoint_path = Path("checkpoints") / "conv_autoencoder.pth"
    codebook_path = Path("codebook.json")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型参数文件：{checkpoint_path}。"
            "请先运行 python train.py 完成训练。"
        )

    # 与 train.py 完全相同，只使用 ToTensor() 将像素缩放到 [0, 1]。
    transform = transforms.ToTensor()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
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
    model.eval()

    integer_counts: dict[int, int] = {}
    latent_element_count = 0

    # 逐批运行编码器和统计频数，不在显存中保留整个训练集的 latent。
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )

            # 本脚本只运行编码器，不调用完整模型，也不会运行解码器。
            latent = model.encode(images)
            quantized_latent = torch.round(latent)

            unique_values, counts = torch.unique(
                quantized_latent,
                return_counts=True,
            )
            unique_values = unique_values.cpu()
            counts = counts.cpu()

            for value, count in zip(unique_values.tolist(), counts.tolist()):
                integer_value = int(value)
                integer_counts[integer_value] = (
                    integer_counts.get(integer_value, 0) + int(count)
                )

            latent_element_count += quantized_latent.numel()

    if latent_element_count == 0:
        raise RuntimeError("训练集没有产生中间数据，无法建立霍夫曼编码表。")
    if sum(integer_counts.values()) != latent_element_count:
        raise RuntimeError("整数频数总和与中间数据元素总数不一致。")

    huffman_codebook = build_huffman_codebook(integer_counts)

    # JSON 对象的键只能是字符串，所以整数在这里会写成字符串。
    # compress.py 读取时会用 int() 把这些键恢复成整数。
    codebook_data = {
        "format": "quantized_conv_autoencoder_huffman_codebook",
        "version": 1,
        "checkpoint": str(checkpoint_path),
        "preprocessing": "torchvision.transforms.ToTensor()",
        "symbols": {
            str(symbol): {
                "count": integer_counts[symbol],
                "code": huffman_codebook[symbol],
            }
            for symbol in sorted(integer_counts)
        },
    }

    with codebook_path.open("w", encoding="utf-8") as file:
        json.dump(
            codebook_data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("\n霍夫曼编码表建立完成：")
    print(f"中间数据元素总数：{latent_element_count:,}")
    print(f"出现的不同整数种类数：{len(integer_counts):,}")
    print(f"编码表保存位置：{codebook_path.resolve()}")


if __name__ == "__main__":
    main()
