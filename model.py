import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """一个用于 CIFAR-10 图像重建的简单卷积 Autoencoder。"""

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # 输入：[batch, 3, 32, 32]；输出：[batch, 16, 16, 16]
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 输入：[batch, 16, 16, 16]；输出：[batch, 32, 8, 8]
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 输入：[batch, 32, 8, 8]；输出：[batch, 64, 4, 4]（latent）
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # 输入：[batch, 64, 4, 4]；输出：[batch, 32, 8, 8]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 输入：[batch, 32, 8, 8]；输出：[batch, 16, 16, 16]
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 输入：[batch, 16, 16, 16]；输出：[batch, 3, 32, 32]
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            # 输入图片在 [0, 1]，因此用 Sigmoid 将重建结果限制在同一范围。
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
