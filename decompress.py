import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import save_image

from compress import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CODEBOOK_PATH,
    FILE_MAGIC,
    FILE_VERSION,
    HEADER_STRUCT,
    load_codebook,
)
from huffman_encoding import decode_integers
from model import ConvAutoencoder


@dataclass
class DecompressionResult:
    """一次解压后，评估和命令行输出需要使用的结果。"""

    decoded_values: list[int]
    quantized_shape: tuple[int, int, int, int]
    reconstructed_tensor: torch.Tensor
    image_width: int
    image_height: int
    valid_bit_count: int
    huffman_data_size: int
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="解压霍夫曼字节流，并使用解码器恢复图片"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="需要解压的二进制文件，例如 test.bin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="重建图片保存位置，例如 reconstructed.png",
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
    return parser.parse_args()


def read_compressed_file(
    input_path: Path,
) -> tuple[tuple[int, int, int], int, int, bytes]:
    """读取并检查 compression v1 文件头和霍夫曼数据。"""
    if not input_path.exists():
        raise FileNotFoundError(f"找不到压缩文件：{input_path}。")
    if not input_path.is_file():
        raise ValueError(f"压缩文件路径不是普通文件：{input_path}。")

    try:
        file_data = input_path.read_bytes()
    except OSError as error:
        raise OSError(f"无法读取压缩文件 {input_path}：{error}") from error

    if len(file_data) < HEADER_STRUCT.size:
        raise ValueError(
            f"压缩文件过短：至少需要 {HEADER_STRUCT.size} 字节的文件头。"
        )

    header_data = file_data[:HEADER_STRUCT.size]
    (
        file_magic,
        file_version,
        latent_channels,
        latent_height,
        latent_width,
        latent_element_count,
        valid_bit_count,
    ) = HEADER_STRUCT.unpack(header_data)

    if file_magic != FILE_MAGIC:
        raise ValueError("文件标识不正确，这不是本项目生成的压缩文件。")
    if file_version != FILE_VERSION:
        raise ValueError(
            f"不支持的压缩文件版本：{file_version}，"
            f"当前程序只支持版本 {FILE_VERSION}。"
        )
    if (
        latent_channels <= 0
        or latent_height <= 0
        or latent_width <= 0
    ):
        raise ValueError("文件头中的中间数据形状不正确。")

    shape_element_count = (
        latent_channels * latent_height * latent_width
    )
    if latent_element_count != shape_element_count:
        raise ValueError(
            "文件头中的中间数据元素数量与形状不一致："
            f"记录数量为 {latent_element_count}，"
            f"形状乘积为 {shape_element_count}。"
        )
    if valid_bit_count <= 0:
        raise ValueError("文件头中的霍夫曼有效位数必须大于 0。")

    huffman_data = file_data[HEADER_STRUCT.size:]

    # compression v1 没有单独重复保存负载长度，因为它可以由有效位数
    # 唯一推导为 ceil(valid_bit_count / 8)，并与文件剩余长度严格核对。
    expected_huffman_bytes = (valid_bit_count + 7) // 8
    if len(huffman_data) < expected_huffman_bytes:
        raise ValueError(
            "霍夫曼数据提前结束："
            f"期望 {expected_huffman_bytes} 字节，"
            f"实际 {len(huffman_data)} 字节。"
        )
    if len(huffman_data) > expected_huffman_bytes:
        raise ValueError(
            "压缩文件在霍夫曼数据之后包含多余内容："
            f"期望 {expected_huffman_bytes} 字节，"
            f"实际 {len(huffman_data)} 字节。"
        )

    latent_shape = (
        latent_channels,
        latent_height,
        latent_width,
    )
    return (
        latent_shape,
        latent_element_count,
        valid_bit_count,
        huffman_data,
    )


def decompress_file(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    codebook_path: Path = DEFAULT_CODEBOOK_PATH,
    device: torch.device | None = None,
) -> DecompressionResult:
    """执行一次完整解压，并返回评估需要的中间结果。"""
    if output_path.exists():
        raise FileExistsError(
            f"输出图片已存在：{output_path}。"
            "为避免覆盖原文件，请更换输出路径。"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型参数文件：{checkpoint_path}。"
            "请确认模型参数路径正确。"
        )
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"找不到霍夫曼编码表：{codebook_path}。"
            "请确认编码表路径正确。"
        )

    (
        latent_shape,
        latent_element_count,
        valid_bit_count,
        huffman_data,
    ) = read_compressed_file(input_path)

    codebook = load_codebook(codebook_path)
    decoded_values = decode_integers(
        encoded_data=huffman_data,
        valid_bit_count=valid_bit_count,
        codebook=codebook,
        expected_value_count=latent_element_count,
    )

    if len(decoded_values) != latent_element_count:
        raise RuntimeError("解码后的整数数量与文件头记录不一致。")

    latent_channels, latent_height, latent_width = latent_shape

    # 编码前的 quantized_latent 是 torch.round() 产生的浮点张量。
    # 因此解码出的整数要恢复为 float32，并补回固定的单图片批次维度。
    quantized_latent = torch.tensor(
        decoded_values,
        dtype=torch.float32,
    ).reshape(
        1,
        latent_channels,
        latent_height,
        latent_width,
    )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = ConvAutoencoder().to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    quantized_latent = quantized_latent.to(device)
    with torch.no_grad():
        # 只调用解码器，不运行编码器，也不调用完整的 model(...)。
        reconstructed_image = model.decode(quantized_latent)

    if reconstructed_image.ndim != 4 or reconstructed_image.size(0) != 1:
        raise RuntimeError("解码器输出的图片批次数不是 1。")

    # 当前训练图片位于 [0, 1]，保存前限制到相同的合法像素范围。
    reconstructed_image = reconstructed_image.clamp(0.0, 1.0)
    reconstructed_on_cpu = reconstructed_image.cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(reconstructed_on_cpu, output_path)

    quantized_shape = (
        1,
        latent_channels,
        latent_height,
        latent_width,
    )
    return DecompressionResult(
        decoded_values=decoded_values,
        quantized_shape=quantized_shape,
        reconstructed_tensor=reconstructed_on_cpu,
        image_width=int(reconstructed_image.size(3)),
        image_height=int(reconstructed_image.size(2)),
        valid_bit_count=valid_bit_count,
        huffman_data_size=len(huffman_data),
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用设备：图形处理器（GPU），{torch.cuda.get_device_name(0)}")
    else:
        print("使用设备：中央处理器（CPU）")

    result = decompress_file(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        codebook_path=args.codebook,
        device=device,
    )

    print("\n解压缩完成：")
    print(
        "解码出的中间数据形状："
        f"{list(result.quantized_shape)}"
    )
    print(f"解码出的整数数量：{len(result.decoded_values):,}")
    print(
        "恢复图片宽度和高度："
        f"{result.image_width} × {result.image_height}"
    )
    print(f"恢复图片保存位置：{result.output_path.resolve()}")
    print(
        "\n说明：恢复该图片需要压缩时使用的同一份模型参数和 "
        "codebook.json。"
    )


if __name__ == "__main__":
    main()
