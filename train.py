import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import ConvAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个简单的卷积 Autoencoder")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数（默认：20）")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="每批图片数量（默认：128）"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Adam 学习率（默认：0.001）"
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

    # CUDA 可用时自动使用 RTX 显卡，否则回退到 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")

    output_dir = Path("outputs")
    checkpoint_dir = Path("checkpoints")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ToTensor() 同时把图片转换成张量，并把像素值缩放到 [0, 1]。
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    # Windows 下 num_workers=0 最直观，也能避免初学时遇到多进程启动问题。
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # 测试集不打乱，所以每个 epoch 都会观察同样的前 8 张图片。
    fixed_images, _ = next(iter(test_loader))
    fixed_images = fixed_images.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_images = 0

        # CIFAR-10 的标签在 Autoencoder 训练中没有用，因此用 _ 忽略。
        for images, _ in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")

            reconstructed, _, _ = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 按图片数量累计，最后得到整个 epoch 的平均 loss。
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

        average_loss = total_loss / total_images
        print(f"Epoch [{epoch:02d}/{args.epochs}]  平均 Loss: {average_loss:.6f}")

        # 保存两行对比图：第一行原图，第二行重建图。
        model.eval()
        with torch.no_grad():
            fixed_reconstructed, _, _ = model(fixed_images)
        comparison = torch.cat(
            (fixed_images.cpu(), fixed_reconstructed.cpu()),
            dim=0,
        )
        image_path = output_dir / f"epoch_{epoch:03d}.png"
        save_image(comparison, image_path, nrow=8)
        print(f"已保存对比图: {image_path}")

    checkpoint_path = checkpoint_dir / "conv_autoencoder.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"训练完成，模型参数已保存到: {checkpoint_path}")


if __name__ == "__main__":
    main()
