# PyTorch 卷积 Autoencoder 入门项目

这个项目用 CIFAR-10 图片演示最基础的图像 Autoencoder 流程：

```text
图像 [batch, 3, 32, 32]
        ↓ Encoder
latent [batch, 64, 4, 4]
        ↓ Decoder
重建图像 [batch, 3, 32, 32]
```

任务目标是让重建图片尽量接近输入图片。CIFAR-10 的类别标签不会参与训练，这不是分类任务。

## 项目结构

```text
.
├── model.py          # ConvAutoencoder 的网络结构
├── train.py          # 下载数据、训练模型并保存每轮重建结果
├── reconstruct.py    # 加载模型，重建 8 张测试图片
├── data/             # CIFAR-10 数据（运行时自动创建）
├── outputs/          # 原图与重建图的对比结果（运行时自动创建）
└── checkpoints/      # 训练好的模型参数（运行时自动创建）
```

## 环境依赖

- Windows、Conda、VS Code
- Python 3.10 或更高版本
- PyTorch
- torchvision
- 支持当前 PyTorch CUDA 版本的 NVIDIA 驱动（使用 GPU 时需要）

可以新建一个 Conda 环境：

```powershell
conda create -n conv-ae python=3.11 -y
conda activate conv-ae
pip install torch torchvision
```

`torch` 和 `torchvision` 应使用同一个官方安装命令一起安装，避免混用不同渠道或不匹配的
版本。由于 CUDA 安装命令会随 PyTorch 版本变化，准备使用 RTX 4060 时，建议通过
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 Windows、
Conda 或 Pip，以及当前推荐的 CUDA 版本。

安装后可检查 PyTorch 是否识别到显卡：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果第一项是 `False`，程序仍能使用 CPU 运行，但会比较慢。此时请通过上面的官方安装
选择器重新安装适合 Windows 的 CUDA 版本。

## 开始训练

请先在终端进入本项目目录并激活 Conda 环境，然后运行：

```powershell
python train.py
```

默认配置为 20 个 epoch、batch size 128、学习率 0.001，数据保存在 `data/`。
程序会自动选择 CUDA 或 CPU，并在终端打印实际使用的设备。每个 epoch 结束后，
`outputs/epoch_XXX.png` 会保存一张两行对比图：第一行是 8 张原图，第二行是对应重建图。
训练完成后，参数保存在 `checkpoints/conv_autoencoder.pth`。

可以用命令行参数修改训练配置：

```powershell
python train.py --epochs 10 --batch-size 64 --learning-rate 0.0005 --data-dir D:\datasets
```

可用参数：

- `--epochs`：训练轮数
- `--batch-size`：每个 batch 的图片数量
- `--learning-rate`：Adam 优化器的学习率
- `--data-dir`：CIFAR-10 的保存位置

第一次运行时 torchvision 会下载 CIFAR-10，需要能够访问网络。

## 加载模型并重建

完成训练后运行：

```powershell
python reconstruct.py
```

脚本会读取 CIFAR-10 测试集的前 8 张图片，打印原图、latent 和重建图片的张量尺寸，
并把对比图保存到 `outputs/reconstruction.png`。如果训练时使用了自定义数据目录，
这里也要传入相同位置：

```powershell
python reconstruct.py --data-dir D:\datasets
```

## 关键概念

### Encoder

Encoder（编码器）用 3 层 `Conv2d` 逐步降低空间尺寸，同时增加通道数：

```text
[batch, 3, 32, 32]
→ [batch, 16, 16, 16]
→ [batch, 32, 8, 8]
→ [batch, 64, 4, 4]
```

它学习从输入图片中提取对重建有用的信息。

### latent

`[batch, 64, 4, 4]` 是图片经过 Encoder 后的中间表示，也叫 latent（潜在表示）。
它的空间尺寸比原图小，后续 Decoder 只根据这份表示来重建图片。你可以把它看作这个
入门项目里的“压缩表示”，但它目前仍是普通浮点张量。

### Decoder

Decoder（解码器）用 3 层 `ConvTranspose2d` 对称地放大空间尺寸，最终恢复到
`[batch, 3, 32, 32]`。隐藏层使用 ReLU，最后使用 Sigmoid，确保输出像素位于
`[0, 1]`，与 `ToTensor()` 处理后的输入范围一致。

### MSELoss

`MSELoss` 计算输入图片与重建图片对应像素之间的均方误差。误差越小，说明重建结果在
像素数值上越接近原图。训练过程通过反向传播调整 Encoder 和 Decoder 的参数，使这个
误差逐渐减小。

## 为什么它还不是真正的 AI Codec

这个项目已经具备 Autoencoder 的核心结构：Encoder 把图像映射为 latent，Decoder 再从
latent 重建图像。但是，latent 还是内存中的连续浮点数，没有被变成可存储、可传输的
压缩比特流，因此它还不是完整的 AI Codec。

真正的 AI Codec 通常还需要：

- **量化（Quantization）**：把连续的 latent 转换成离散符号。
- **熵模型（Entropy Model）**：估计离散符号的概率，为码率建模。
- **熵编码（Entropy Coding）**：用算术编码、ANS 等方法生成真正的压缩比特流。
- **率失真优化（Rate-Distortion Optimization）**：同时优化码率和重建质量，而不只是
  最小化 MSE。常见目标形式是 `loss = distortion + λ × rate`。

因此，本项目适合先理解“图像 → latent → 重建图像”的主干流程，再继续学习神经网络
图像压缩中的量化、概率建模和比特流编码。
