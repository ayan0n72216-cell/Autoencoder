import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from compress import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CODEBOOK_PATH,
    compress_image,
)
from decompress import decompress_file


DEFAULT_OUTPUT_DIR = Path("outputs") / "codec_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估一次完整的图片压缩、解压和重建流程"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="需要评估的一张图片",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="评估结果目录（默认：outputs/codec_evaluation）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="模型参数文件（默认：checkpoints/conv_autoencoder.pth）",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=DEFAULT_CODEBOOK_PATH,
        help="霍夫曼编码表（默认：codebook.json）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中本次评估的四个结果文件",
    )
    return parser.parse_args()


def calculate_mse_and_psnr(
    input_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor,
) -> tuple[float, float]:
    """计算 [0, 1] 图片的均方误差和峰值信噪比。"""
    if input_tensor.shape != reconstructed_tensor.shape:
        raise ValueError(
            "输入编码器的图片与重建图片尺寸不一致："
            f"输入为 {list(input_tensor.shape)}，"
            f"重建为 {list(reconstructed_tensor.shape)}。"
            "当前压缩预处理不会缩放图片，请使用解码器能够恢复为"
            "相同尺寸的输入图片。"
        )

    mse = torch.mean(
        (input_tensor.float() - reconstructed_tensor.float()) ** 2
    ).item()

    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10(1.0 / mse)

    return mse, psnr


def calculate_bits_per_pixel(
    bit_count: int,
    image_height: int,
    image_width: int,
) -> float:
    """按空间像素数量计算每像素位数。"""
    if bit_count < 0:
        raise ValueError("位数不能小于 0。")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("图片高度和宽度必须大于 0。")

    # 一个空间像素包含三个颜色分量，因此分母不乘 RGB 通道数。
    spatial_pixel_count = image_height * image_width
    return bit_count / spatial_pixel_count


def calculate_compression_ratio(
    original_size_bytes: int,
    compressed_size_bytes: int,
) -> float:
    """计算原始大小除以压缩文件大小的比值。"""
    if original_size_bytes < 0:
        raise ValueError("原始数据大小不能小于 0。")
    if compressed_size_bytes <= 0:
        raise ValueError("压缩文件大小必须大于 0。")
    return original_size_bytes / compressed_size_bytes


def save_comparison_image(
    input_tensor: torch.Tensor,
    reconstructed_tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """保存带英文标题的输入与重建图片对比图。"""
    if input_tensor.shape != reconstructed_tensor.shape:
        raise ValueError(
            "无法生成对比图：输入图片与重建图片尺寸不一致。"
        )
    if (
        input_tensor.ndim != 4
        or input_tensor.size(0) != 1
        or input_tensor.size(1) != 3
    ):
        raise ValueError("对比图需要形状为 [1, 3, H, W] 的图片张量。")

    input_image = to_pil_image(input_tensor[0].clamp(0.0, 1.0))
    reconstructed_image = to_pil_image(
        reconstructed_tensor[0].clamp(0.0, 1.0)
    )

    image_width, image_height = input_image.size
    label_height = 24
    # CIFAR-10 图片只有 32 像素宽，面板留出足够空间显示英文标题。
    panel_width = max(image_width, 160)
    canvas = Image.new(
        "RGB",
        (panel_width * 2, image_height + label_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)

    input_left = (panel_width - image_width) // 2
    reconstructed_left = (
        panel_width + (panel_width - image_width) // 2
    )
    canvas.paste(input_image, (input_left, label_height))
    canvas.paste(
        reconstructed_image,
        (reconstructed_left, label_height),
    )
    draw.text((8, 6), "Input to encoder", fill="black")
    draw.text(
        (panel_width + 8, 6),
        "Reconstructed",
        fill="black",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_metrics_json(
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    """将指标保存为可重新读取的标准 JSON 文件。"""
    json_metrics = dict(metrics)

    # 标准 JSON 没有无穷大数值。当 MSE 为 0 时，用清楚的字符串保存。
    psnr = json_metrics.get("psnr_db")
    if isinstance(psnr, float) and math.isinf(psnr):
        json_metrics["psnr_db"] = "Infinity"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            json_metrics,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


def synchronize_cuda(device: torch.device) -> None:
    """CUDA 异步执行时，用同步保证耗时统计准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_codec(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    codebook_path: Path = DEFAULT_CODEBOOK_PATH,
    overwrite: bool = False,
) -> dict[str, object]:
    """执行完整 Codec 评估并返回全部指标。"""
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入图片：{input_path}。")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型参数文件：{checkpoint_path}。"
        )
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"找不到霍夫曼编码表：{codebook_path}。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    compressed_path = output_dir / "compressed.bin"
    reconstructed_path = output_dir / "reconstructed.png"
    comparison_path = output_dir / "comparison.png"
    metrics_path = output_dir / "metrics.json"
    output_paths = (
        compressed_path,
        reconstructed_path,
        comparison_path,
        metrics_path,
    )

    protected_paths = {
        input_path.resolve(),
        checkpoint_path.resolve(),
        codebook_path.resolve(),
    }
    for path in output_paths:
        if path.resolve() in protected_paths:
            raise ValueError(
                f"评估输出路径不能与输入或依赖文件相同：{path}。"
            )

    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        existing_text = "、".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"以下评估结果已经存在：{existing_text}。"
            "请更换 --output-dir，或明确添加 --overwrite。"
        )

    if overwrite:
        for path in existing_paths:
            if not path.is_file():
                raise ValueError(f"不能覆盖非文件路径：{path}。")
            path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 压缩耗时包含：读取输入图片、预处理、加载模型和编码表、
    # 编码器推理、量化、霍夫曼编码以及写入 compressed.bin。
    synchronize_cuda(device)
    compression_start = time.perf_counter()
    compression_result = compress_image(
        input_path=input_path,
        output_path=compressed_path,
        checkpoint_path=checkpoint_path,
        codebook_path=codebook_path,
        device=device,
    )
    synchronize_cuda(device)
    compression_time_ms = (
        time.perf_counter() - compression_start
    ) * 1000.0

    # 解压耗时包含：读取 .bin 和编码表、霍夫曼解码、加载模型、
    # 解码器推理以及写入 reconstructed.png。
    synchronize_cuda(device)
    decompression_start = time.perf_counter()
    decompression_result = decompress_file(
        input_path=compressed_path,
        output_path=reconstructed_path,
        checkpoint_path=checkpoint_path,
        codebook_path=codebook_path,
        device=device,
    )
    synchronize_cuda(device)
    decompression_time_ms = (
        time.perf_counter() - decompression_start
    ) * 1000.0

    if (
        compression_result.quantized_values
        != decompression_result.decoded_values
    ):
        raise RuntimeError(
            "霍夫曼解码得到的整数与压缩前量化整数不一致，"
            "已停止画质评估。"
        )
    if (
        compression_result.quantized_shape
        != decompression_result.quantized_shape
    ):
        raise RuntimeError("压缩前后的量化中间数据形状不一致。")

    input_tensor = compression_result.input_tensor
    reconstructed_tensor = decompression_result.reconstructed_tensor
    mse, psnr = calculate_mse_and_psnr(
        input_tensor,
        reconstructed_tensor,
    )

    image_width = compression_result.image_width
    image_height = compression_result.image_height
    spatial_pixel_count = image_height * image_width

    huffman_bits_per_pixel = calculate_bits_per_pixel(
        compression_result.valid_bit_count,
        image_height,
        image_width,
    )
    complete_bits_per_pixel = calculate_bits_per_pixel(
        compression_result.compressed_file_size * 8,
        image_height,
        image_width,
    )

    original_file_compression_ratio = calculate_compression_ratio(
        compression_result.input_file_size,
        compression_result.compressed_file_size,
    )
    raw_pixel_data_size = spatial_pixel_count * 3
    raw_pixel_compression_ratio = calculate_compression_ratio(
        raw_pixel_data_size,
        compression_result.compressed_file_size,
    )

    save_comparison_image(
        input_tensor,
        reconstructed_tensor,
        comparison_path,
    )

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    metrics: dict[str, object] = {
        "input_path": str(input_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "codebook_path": str(codebook_path.resolve()),
        "input_width": image_width,
        "input_height": image_height,
        "spatial_pixel_count": spatial_pixel_count,
        "original_file_size_bytes": compression_result.input_file_size,
        "raw_pixel_data_size_bytes": raw_pixel_data_size,
        "header_size_bytes": compression_result.header_size,
        "huffman_data_size_bytes": compression_result.huffman_data_size,
        "compressed_file_size_bytes": (
            compression_result.compressed_file_size
        ),
        "huffman_valid_bits": compression_result.valid_bit_count,
        "huffman_bits_per_pixel": huffman_bits_per_pixel,
        "complete_file_bits_per_pixel": complete_bits_per_pixel,
        "original_file_compression_ratio": (
            original_file_compression_ratio
        ),
        "raw_pixel_data_compression_ratio": (
            raw_pixel_compression_ratio
        ),
        "mse": mse,
        "psnr_db": psnr,
        "compression_time_ms": compression_time_ms,
        "decompression_time_ms": decompression_time_ms,
        "quantized_latent_shape": list(
            compression_result.quantized_shape
        ),
        "quantized_element_count": (
            compression_result.latent_element_count
        ),
        "huffman_lossless_verified": True,
        "device": device_name,
        "compression_timing_scope": (
            "读取图片、预处理、加载模型和编码表、编码器推理、"
            "量化、霍夫曼编码、写入 compressed.bin"
        ),
        "decompression_timing_scope": (
            "读取压缩文件和编码表、霍夫曼解码、加载模型、"
            "解码器推理、写入 reconstructed.png"
        ),
        "compressed_path": str(compressed_path.resolve()),
        "reconstructed_path": str(reconstructed_path.resolve()),
        "comparison_path": str(comparison_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
    }
    save_metrics_json(metrics, metrics_path)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate_codec(
        input_path=args.input,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        codebook_path=args.codebook,
        overwrite=args.overwrite,
    )

    psnr = metrics["psnr_db"]
    if isinstance(psnr, float) and math.isinf(psnr):
        psnr_text = "正无穷"
    else:
        psnr_text = f"{float(psnr):.4f} 分贝"

    print("\nCodec 评估完成：")
    print(
        "输入编码器的图片尺寸："
        f"{metrics['input_width']} × {metrics['input_height']}"
    )
    print(
        "原始图片文件大小："
        f"{metrics['original_file_size_bytes']:,} 字节"
    )
    print(f"文件头大小：{metrics['header_size_bytes']:,} 字节")
    print(
        "霍夫曼数据大小："
        f"{metrics['huffman_data_size_bytes']:,} 字节"
    )
    print(
        "完整压缩文件大小："
        f"{metrics['compressed_file_size_bytes']:,} 字节"
    )
    print(
        "霍夫曼有效位数："
        f"{metrics['huffman_valid_bits']:,} 位"
    )
    print(
        "纯霍夫曼每像素位数："
        f"{metrics['huffman_bits_per_pixel']:.6f} 位/像素"
    )
    print(
        "完整文件每像素位数："
        f"{metrics['complete_file_bits_per_pixel']:.6f} 位/像素"
    )
    print(
        "原文件压缩比："
        f"{metrics['original_file_compression_ratio']:.6f}"
    )
    print(
        "原始像素数据压缩比："
        f"{metrics['raw_pixel_data_compression_ratio']:.6f}"
    )
    print(f"平均平方误差：{metrics['mse']:.8f}")
    print(f"峰值信噪比：{psnr_text}")
    print(f"压缩耗时：{metrics['compression_time_ms']:.3f} 毫秒")
    print(f"解压耗时：{metrics['decompression_time_ms']:.3f} 毫秒")
    print("霍夫曼中间数据无损校验：通过")
    print(f"压缩文件：{metrics['compressed_path']}")
    print(f"重建图片：{metrics['reconstructed_path']}")
    print(f"对比图片：{metrics['comparison_path']}")
    print(f"指标文件：{metrics['metrics_path']}")
    print(
        "\n耗时说明：压缩和解压耗时均为一次命令的端到端阶段耗时，"
        "包含各自的文件读取、模型参数加载、推理和结果文件写入；"
        "不包含 comparison.png 与 metrics.json 的生成时间。"
    )
    print(
        "原文件压缩比受 PNG/JPEG 既有压缩影响；"
        "原始像素数据压缩比使用 宽×高×3 字节作为未压缩基准。"
    )


if __name__ == "__main__":
    main()
