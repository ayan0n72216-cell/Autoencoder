import torch
from torch import nn


def quantize(latent: torch.Tensor, training: bool) -> torch.Tensor:
    """训练时用均匀噪声近似量化，测试时执行真正的四舍五入。"""
    if training:
        noise = torch.empty_like(latent).uniform_(-0.5, 0.5)
        return latent + noise

    return torch.round(latent)


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

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """只运行编码器，返回量化前的中间数据。"""
        return self.encoder(images)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        # self.training 会随 model.train() / model.eval() 自动切换。
        quantized_latent = quantize(latent, training=self.training)
        reconstructed = self.decoder(quantized_latent)
        return reconstructed, latent, quantized_latent
