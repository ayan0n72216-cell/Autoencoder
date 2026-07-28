import argparse
import json
import struct
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from huffman_encoding import encode_integers
from model import ConvAutoencoder


# 文件头使用大端字节序，字段依次为：
# 8 字节文件标识、1 字节版本、三个 4 字节 latent 尺寸、
# 8 字节元素总数、8 字节霍夫曼有效位数。
FILE_MAGIC = b"QCAEHUFF"
FILE_VERSION = 1
HEADER_STRUCT = struct.Struct(">8sBIIIQQ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只运行编码器并把一张图片压缩为霍夫曼字节流"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="需要压缩的一张图片",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="压缩文件保存位置，例如 compressed.bin",
    )
    return parser.parse_args()


def load_codebook(codebook_path: Path) -> dict[int, str]:
    """读取 JSON 编码表，并把字符串键恢复成整数。"""
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"找不到霍夫曼编码表：{codebook_path}。"
            "请先运行 python build_codebook.py。"
        )

    try:
        with codebook_path.open("r", encoding="utf-8") as file:
            codebook_data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"无法读取霍夫曼编码表 {codebook_path}：{error}"
        ) from error

    if (
        codebook_data.get("format")
        != "quantized_conv_autoencoder_huffman_codebook"
        or codebook_data.get("version") != 1
    ):
        raise ValueError(
            "codebook.json 的格式或版本不受支持，"
            "请重新运行 build_codebook.py。"
        )

    symbols = codebook_data.get("symbols")
    if not isinstance(symbols, dict) or not symbols:
        raise ValueError(
            "codebook.json 中没有有效的整数编码，"
            "请重新运行 build_codebook.py。"
        )

    codebook: dict[int, str] = {}
    try:
        for symbol_text, symbol_data in symbols.items():
            # JSON 键是字符串，压缩前必须恢复为真正的整数。
            integer_symbol = int(symbol_text)
            code = symbol_data["code"]
            if not isinstance(code, str):
                raise TypeError
            codebook[integer_symbol] = code
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "codebook.json 中的整数或二进制编码格式不正确，"
            "请重新运行 build_codebook.py。"
        ) from error

    return codebook


def write_compressed_file(
    output_path: Path,
    latent_shape: tuple[int, int, int],
    element_count: int,
    valid_bit_count: int,
    huffman_data: bytes,
) -> int:
    """写入固定文件头和霍夫曼字节数据，返回文件头字节数。"""
    if output_path.exists():
        raise FileExistsError(
            f"输出文件已存在：{output_path}。"
            "为避免覆盖原文件，请更换 --output 路径。"
        )

    channels, height, width = latent_shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("量化后的中间数据形状必须全部大于 0。")
    if element_count != channels * height * width:
        raise ValueError("中间数据元素数量与中间数据形状不一致。")

    expected_data_bytes = (valid_bit_count + 7) // 8
    if valid_bit_count <= 0:
        raise ValueError("霍夫曼编码的有效位数必须大于 0。")
    if len(huffman_data) != expected_data_bytes:
        raise ValueError("霍夫曼数据字节数与有效位数不一致。")

    header = HEADER_STRUCT.pack(
        FILE_MAGIC,
        FILE_VERSION,
        channels,
        height,
        width,
        element_count,
        valid_bit_count,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 xb 模式再次阻止覆盖，避免检查后文件恰好被其他程序创建。
    with output_path.open("xb") as file:
        file.write(header)
        file.write(huffman_data)

    return len(header)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path("checkpoints") / "conv_autoencoder.pth"
    codebook_path = Path("codebook.json")

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入图片：{args.input}。")
    if not args.input.is_file():
        raise ValueError(f"输入路径不是文件：{args.input}。")
    if args.output.exists():
        raise FileExistsError(
            f"输出文件已存在：{args.output}。"
            "为避免覆盖原文件，请更换 --output 路径。"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型参数文件：{checkpoint_path}。"
            "请先运行 python train.py 完成训练。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用设备：图形处理器（GPU），{torch.cuda.get_device_name(0)}")
    else:
        print("使用设备：中央处理器（CPU）")

    codebook = load_codebook(codebook_path)

    # 先转换为三通道彩色图片，再使用训练时相同的 ToTensor()。
    with Image.open(args.input) as source_image:
        rgb_image = source_image.convert("RGB")
        image_width, image_height = rgb_image.size
        image = transforms.ToTensor()(rgb_image).unsqueeze(0)

    # 本脚本一次只允许一张图片，所以批次数必须严格等于 1。
    if image.ndim != 4 or image.size(0) != 1:
        raise RuntimeError("输入模型的图片批次数必须为 1。")

    image = image.to(device)
    model = ConvAutoencoder().to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # 只调用编码器，不调用 model(image)，因此解码器不会运行。
        latent = model.encode(image)
        quantized_latent = torch.round(latent)

    if quantized_latent.ndim != 4 or quantized_latent.size(0) != 1:
        raise RuntimeError("量化后的中间数据批次数必须为 1。")

    quantized_shape = tuple(quantized_latent.shape)
    _, latent_channels, latent_height, latent_width = quantized_shape
    latent_element_count = quantized_latent.numel()

    # torch.round() 后的数值都是整数，转为中央处理器上的整数列表后编码。
    integer_values = (
        quantized_latent.to(dtype=torch.int64).flatten().cpu().tolist()
    )
    huffman_data, valid_bit_count = encode_integers(
        integer_values,
        codebook,
    )

    header_size = write_compressed_file(
        output_path=args.output,
        latent_shape=(latent_channels, latent_height, latent_width),
        element_count=latent_element_count,
        valid_bit_count=valid_bit_count,
        huffman_data=huffman_data,
    )

    input_file_size = args.input.stat().st_size
    huffman_data_size = len(huffman_data)
    compressed_file_size = args.output.stat().st_size

    # 图像压缩通常按空间像素数量计算“位/像素”，一个空间像素包含
    # 三个颜色分量，因此分母不再乘 RGB 颜色通道数。
    spatial_pixel_count = image.size(0) * image_height * image_width
    huffman_bits_per_pixel = valid_bit_count / spatial_pixel_count
    complete_file_bits_per_pixel = (
        compressed_file_size * 8 / spatial_pixel_count
    )

    print("\n图片压缩完成：")
    print(f"输入图片文件大小：{input_file_size:,} 字节")
    print(
        "输入模型后的图片宽度和高度："
        f"{image_width} × {image_height}"
    )
    print(f"量化后的中间数据形状：{list(quantized_shape)}")
    print(f"中间数据元素数量：{latent_element_count:,}")
    print(f"霍夫曼编码的有效位数：{valid_bit_count:,} 位")
    print(f"霍夫曼数据占用字节数：{huffman_data_size:,} 字节")
    print(f"文件头占用字节数：{header_size:,} 字节")
    print(f"完整压缩文件大小：{compressed_file_size:,} 字节")
    print(
        "纯霍夫曼数据平均每个原图像素使用的位数："
        f"{huffman_bits_per_pixel:.6f} 位/像素"
    )
    print(
        "完整压缩文件平均每个原图像素使用的位数："
        f"{complete_file_bits_per_pixel:.6f} 位/像素"
    )
    print(f"压缩文件保存位置：{args.output.resolve()}")

    print(
        "\n说明：compressed.bin 没有重复保存整个霍夫曼编码表，"
        "它依赖生成该编码表时使用的同一份模型参数和 codebook.json。"
    )
    print(
        "PNG 和 JPEG 文件本身已经经过传统压缩，所以输入文件大小与"
        "本项目生成的文件大小不能直接用于公平比较。"
    )


if __name__ == "__main__":
    main()
