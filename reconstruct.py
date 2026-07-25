import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import ConvAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 Autoencoder 重建图片")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="CIFAR-10 数据保存位置（默认：data）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    checkpoint_path = Path("checkpoints") / "conv_autoencoder.pth"
    output_path = Path("outputs") / "reconstruction.png"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型参数：{checkpoint_path}。请先运行 python train.py。"
        )

    model = ConvAutoencoder().to(device)
    # map_location 保证模型既能在 CUDA 上加载，也能在只有 CPU 时加载。
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # 标签不是重建所需的信息，因此同样忽略。
    original_images, _ = next(iter(test_loader))
    original_images = original_images.to(device)

    # eval() 关闭训练模式；no_grad() 避免保存梯度，推理更省显存。
    with torch.no_grad():
        reconstructed_images, latent = model(original_images)

    print(f"原图片张量尺寸: {list(original_images.shape)}")
    print(f"latent 张量尺寸: {list(latent.shape)}")
    print(f"重建图片张量尺寸: {list(reconstructed_images.shape)}")

    # nrow=8 使前 8 张原图位于第一行，后 8 张重建图位于第二行。
    comparison = torch.cat(
        (original_images.cpu(), reconstructed_images.cpu()),
        dim=0,
    )
    save_image(comparison, output_path, nrow=8)
    print(f"重建对比图已保存到: {output_path}")


if __name__ == "__main__":
    main()
