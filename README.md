# PyTorch 量化卷积 Autoencoder Codec

这是一个面向初学者的 CIFAR-10 图像压缩项目。项目从卷积 Autoencoder 出发，已经实现
训练、量化、公共霍夫曼编码表、真实二进制压缩、解压重建，以及压缩质量评估的完整流程。

```text
压缩：图片 → Encoder → latent → 四舍五入 → Huffman 编码 → .bin
解压：.bin → Huffman 解码 → 量化 latent → Decoder → 重建图片
```

仓库地址：[ayan0n72216-cell/Autoencoder](https://github.com/ayan0n72216-cell/Autoencoder)

## 项目特点

- 使用 3 层卷积编码器和 3 层转置卷积解码器处理 CIFAR-10 RGB 图片。
- 训练时给 latent 添加 `[-0.5, 0.5]` 均匀噪声，近似不可导的量化过程。
- 推理、统计和压缩时使用 `torch.round()` 得到真正的整数 latent。
- 遍历完整 CIFAR-10 训练集，建立稳定的公共霍夫曼编码表。
- 将霍夫曼位流按每 8 位打包为真实字节，而不是保存 `"0101"` 文本。
- 使用带固定文件头的 `QCAEHUFF` v1 二进制格式。
- 支持单图压缩、解压、全测试集理论熵评估和单图 Codec 端到端评估。
- 自动选择 CUDA 或 CPU；只有 CPU 时也可以运行。
- 对缺失文件、未知整数、可检测的位流结构损坏、错误文件头和覆盖已有文件等情况给出明确错误。

## 项目结构

`main` 分支包含以下文件：

```text
.
├── model.py              # 网络结构、训练/推理量化、独立 encode/decode 接口
├── train.py              # 训练模型并保存每轮重建对比图和最终权重
├── reconstruct.py        # 从 CIFAR-10 测试集取一批图片进行重建展示
├── evaluate.py           # 整个测试集的重建质量与理论熵大小评估
├── build_codebook.py     # 遍历训练集，生成公共 codebook.json
├── huffman_encoding.py   # 霍夫曼树、字节编码和字节解码
├── compress.py           # 图片 → 量化 latent → 霍夫曼字节流 → .bin
├── decompress.py         # .bin → 整数 latent → Decoder → 重建图片
├── evaluate_codec.py     # 单图真实压缩—解压—画质/码率/耗时评估
├── test_compression.py   # 霍夫曼编码和二进制文件头的 unittest 测试
├── testimage.py          # 从 CIFAR-10 测试集导出 test.png 示例图片
└── README.md
```

程序运行后还会使用或生成：

```text
data/                              # CIFAR-10 数据集
checkpoints/conv_autoencoder.pth   # 训练好的模型参数
codebook.json                      # 公共霍夫曼编码表
outputs/                           # 训练、重建和评估输出
```

`train.py`、`reconstruct.py`、`evaluate.py`、`build_codebook.py` 和 `testimage.py`
加载 CIFAR-10 时都使用 `download=True`：若 `data/` 中没有数据集，运行这些脚本会尝试自动下载。

## 环境依赖

- Python 3.10 或更高版本
- PyTorch
- torchvision
- Pillow
- 可选：支持当前 PyTorch CUDA 版本的 NVIDIA 显卡和驱动

推荐在独立 Conda 环境中安装：

```powershell
conda create -n conv-ae python=3.11 -y
conda activate conv-ae
pip install torch torchvision pillow
```

`torch` 与 `torchvision` 应使用同一个官方安装命令安装。需要 CUDA 时，请通过
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择与系统和显卡匹配的版本。

检查 PyTorch 是否识别显卡：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 快速开始：训练到真实 Codec 评估

### 1. 训练模型

```powershell
python train.py
```

默认训练配置：

- 20 个 epoch
- batch size 128
- Adam 学习率 0.001
- CIFAR-10 数据目录为 `data/`

每轮结束后会生成 `outputs/epoch_XXX.png`。训练完成后，模型参数保存为：

```text
checkpoints/conv_autoencoder.pth
```

自定义训练参数：

```powershell
python train.py --epochs 10 --batch-size 64 --learning-rate 0.0005 --data-dir D:\datasets
```

如果已经有与当前网络结构匹配的模型参数，可以跳过训练。

### 2. 建立公共霍夫曼编码表

```powershell
python build_codebook.py
```

这个脚本会：

1. 加载 `checkpoints/conv_autoencoder.pth`。
2. 切换到 `model.eval()` 并关闭梯度。
3. 遍历完整 CIFAR-10 训练集。
4. 只运行编码器并使用 `torch.round()` 量化 latent。
5. 统计每个整数的出现次数。
6. 建立稳定的霍夫曼编码并保存到 `codebook.json`。

可指定批大小和数据目录：

```powershell
python build_codebook.py --batch-size 256 --data-dir D:\datasets
```

模型参数发生变化后，应重新生成 `codebook.json`。

### 3. 准备一张测试图片

可以从 CIFAR-10 测试集导出第一张图片：

```powershell
python testimage.py
```

脚本会生成 `test.png`，并打印图片模式、尺寸、张量形状和类别编号。也可以使用自己的
RGB 图片，但当前预处理只执行 `RGB → ToTensor()`，不会自动缩放。

模型针对 32×32 图片训练。若希望压缩前后的空间尺寸一致，输入宽高应为 8 的倍数；
最推荐直接使用 32×32 图片。

### 4. 压缩图片

```powershell
python compress.py --input test.png --output test.bin
```

压缩流程：

```text
读取图片
→ 转为 RGB
→ ToTensor()，像素范围变为 [0, 1]
→ model.encode()
→ torch.round()
→ 展平为整数序列
→ 使用 codebook.json 进行霍夫曼编码
→ 写入固定文件头和霍夫曼字节
```

`compress.py` 一次只处理一张图片。输出文件已存在时会拒绝覆盖。

### 5. 解压并恢复图片

```powershell
python decompress.py --input test.bin --output reconstructed.png
```

解压流程：

```text
读取并检查文件头
→ 读取 codebook.json
→ 霍夫曼解码整数序列
→ 恢复 [1, C, H, W] 浮点 latent
→ model.decode()
→ 限制到 [0, 1]
→ 保存重建图片
```

可以显式指定模型参数和编码表：

```powershell
python decompress.py `
  --input test.bin `
  --output reconstructed.png `
  --checkpoint checkpoints\conv_autoencoder.pth `
  --codebook codebook.json
```

`.bin` 不包含模型参数或整份编码表，因此解压必须使用压缩时对应的模型参数和
`codebook.json`。

### 6. 进行一次完整 Codec 评估

```powershell
python evaluate_codec.py --input test.png
```

默认输出目录为 `outputs/codec_evaluation/`，其中包含：

```text
compressed.bin      # 真实压缩文件
reconstructed.png   # 解压重建图片
comparison.png      # Input to encoder / Reconstructed 对比图
metrics.json        # 可再次读取的完整数值指标
```

自定义路径：

```powershell
python evaluate_codec.py `
  --input test.png `
  --output-dir outputs\codec_evaluation `
  --checkpoint checkpoints\conv_autoencoder.pth `
  --codebook codebook.json
```

为防止误覆盖，只要四个目标文件中有一个已经存在，程序就会停止。确认需要覆盖时使用：

```powershell
python evaluate_codec.py --input test.png --overwrite
```

评估器还会逐项检查压缩前的量化整数与霍夫曼解码后的整数是否完全相同。只有霍夫曼
往返无损时，才会继续计算图片质量指标。

## 其他常用功能

### 观察一批测试图片的重建结果

```powershell
python reconstruct.py
```

脚本从 CIFAR-10 测试集随机取一批 8 张图片，打印原图、latent、量化 latent 和重建图的
形状，并将对比图保存为 `outputs/reconstruction.png`。

指定数据目录：

```powershell
python reconstruct.py --data-dir D:\datasets
```

### 评估完整 CIFAR-10 测试集

```powershell
python evaluate.py
```

`evaluate.py` 遍历全部测试图片并统计：

- 全局平均重建均方误差（MSE）
- 峰值信噪比（PSNR）
- 量化整数的完整频数
- 不同整数种类数
- 平均每个 latent 元素的理论位数（经验熵）
- 理论总位数、总字节数和每空间像素位数

这里的存储大小是根据整数概率计算的理论估计，不会调用霍夫曼编码，也不会生成压缩文件。
需要真实 `.bin` 大小和单图画质时，应使用 `evaluate_codec.py`。

可指定批大小和数据目录：

```powershell
python evaluate.py --batch-size 256 --data-dir D:\datasets
```

## 网络结构与量化

### Encoder

```text
[batch, 3, 32, 32]
→ Conv2d + ReLU → [batch, 16, 16, 16]
→ Conv2d + ReLU → [batch, 32, 8, 8]
→ Conv2d + ReLU → [batch, 64, 4, 4]
```

### Quantization

训练模式：

```python
quantized_latent = latent + Uniform(-0.5, 0.5)
```

测试、统计和压缩模式：

```python
quantized_latent = torch.round(latent)
```

训练时使用连续噪声，使梯度仍能从解码器传回编码器；实际压缩时必须使用真正的整数符号。

### Decoder

```text
[batch, 64, 4, 4]
→ ConvTranspose2d + ReLU → [batch, 32, 8, 8]
→ ConvTranspose2d + ReLU → [batch, 16, 16, 16]
→ ConvTranspose2d + Sigmoid → [batch, 3, 32, 32]
```

最后的 `Sigmoid` 将输出限制在 `[0, 1]`，与 `ToTensor()` 后的输入范围一致。

## 霍夫曼编码表

`codebook.json` 保存编码表版本、模型参数路径、预处理说明，以及每个整数的训练集频数和
二进制编码。JSON 键会保存为字符串，读取后再恢复成整数。

简化示例：

```json
{
  "format": "quantized_conv_autoencoder_huffman_codebook",
  "version": 1,
  "symbols": {
    "0": {
      "count": 123,
      "code": "00"
    }
  }
}
```

霍夫曼实现具有以下行为：

- 相同频数统计稳定生成相同编码表。
- 支持负整数、零和正整数。
- 只有一种整数时使用非空编码 `0`。
- 最后不足 8 位时在右侧补零，补零不计入有效位数。
- 解码时检查补零、前缀规则、数据长度和元素数量。
- 压缩时遇到编码表中不存在的整数会停止，并提示重新建立编码表。

## `.bin` 文件格式

文件头使用大端字节序：

```python
struct.Struct(">8sBIIIQQ")
```

固定文件头共 37 字节：

| 偏移 | 字节数 | 字段 | 含义 |
|---:|---:|---|---|
| 0 | 8 | 文件标识 | 固定为 `QCAEHUFF` |
| 8 | 1 | 文件版本 | 当前为 `1` |
| 9 | 4 | latent 通道数 | `C` |
| 13 | 4 | latent 高度 | `H` |
| 17 | 4 | latent 宽度 | `W` |
| 21 | 8 | 元素总数 | 必须等于 `C × H × W` |
| 29 | 8 | 有效位数 | 不包含末字节补零 |
| 37 | 不定 | 霍夫曼数据 | 真正的打包字节流 |

霍夫曼负载字节数可以由 `ceil(有效位数 / 8)` 唯一确定。解压时会检查文件剩余长度是否
完全一致，并拒绝错误标识、未知版本、形状冲突、截断数据和多余数据。

当前格式没有保存校验和，因此这些检查针对文件结构、长度、补零、霍夫曼前缀和元素数量；
如果比特翻转后仍恰好构成合法码字，程序不一定能够发现。需要传输级完整性保证时，应在容器外
另行校验文件哈希。

## Codec 评估指标

`evaluate_codec.py` 会输出并写入 `metrics.json`：

- **原始图片文件大小**：输入 PNG/JPEG 在磁盘上的实际字节数。
- **文件头大小**：当前固定为 37 字节。
- **霍夫曼数据大小**：打包后的负载字节数，末字节可能包含补零。
- **霍夫曼有效位数**：真正参与编码的位数，不包含补零。
- **纯霍夫曼每像素位数**：`有效位数 / (高度 × 宽度)`。
- **完整文件每像素位数**：`.bin 字节数 × 8 / (高度 × 宽度)`。
- **原文件压缩比**：`输入图片文件字节数 / .bin 字节数`。
- **原始像素数据压缩比**：`高度 × 宽度 × 3 / .bin 字节数`。
- **MSE**：输入编码器的 `[0, 1]` 张量与解压重建张量的平均平方误差。
- **PSNR**：`10 × log10(1 / MSE)`；MSE 为 0 时是正无穷。
- **压缩/解压耗时**：包含文件读取、模型和编码表加载、推理以及对应结果写入。

每像素位数的分母不乘颜色通道数，因为图像压缩通常按空间像素数量统计。

PNG 和 JPEG 本身已经经过传统压缩，所以“原文件压缩比”不适合单独作为模型优劣依据；
应结合原始像素数据压缩比、每像素位数、MSE 和 PSNR 一起观察。

## 测试

先检查语法：

```powershell
python -m py_compile `
  model.py `
  train.py `
  reconstruct.py `
  evaluate.py `
  build_codebook.py `
  huffman_encoding.py `
  compress.py `
  decompress.py `
  evaluate_codec.py `
  test_compression.py `
  testimage.py
```

运行仓库中的单元测试：

```powershell
python test_compression.py
```

该测试覆盖正整数、负整数和零、非整字节补零、单符号编码、空输入、未知整数、稳定编码表、
真实字节输出、文件头大小、元素数量以及拒绝覆盖已有文件。

## 可复用 Python 接口

除命令行之外，其他 Python 程序也可以复用核心函数：

```python
from pathlib import Path

from compress import compress_image
from decompress import decompress_file

compression_result = compress_image(
    input_path=Path("test.png"),
    output_path=Path("test.bin"),
)

decompression_result = decompress_file(
    input_path=Path("test.bin"),
    output_path=Path("reconstructed.png"),
)
```

`CompressionResult` 会提供输入张量、量化整数、latent 形状、有效位数和文件大小；
`DecompressionResult` 会提供解码整数、重建张量、输出尺寸和霍夫曼负载信息。

## 常见问题

### 找不到模型参数

先运行：

```powershell
python train.py
```

确认存在 `checkpoints/conv_autoencoder.pth`。

### 找不到 `codebook.json`

运行：

```powershell
python build_codebook.py
```

### 出现“整数不在霍夫曼编码表中”

当前图片产生了训练码表未覆盖的整数。请确认模型参数与编码表配套，然后使用同一模型重新运行
`build_codebook.py`。程序不会删除、截断或替换未知整数。

### 输出文件已经存在

`compress.py` 和 `decompress.py` 默认拒绝覆盖，请换一个输出路径。`evaluate_codec.py` 只有在
明确传入 `--overwrite` 时才会覆盖自己的四个目标文件。

### Codec 评估提示输入与重建尺寸不同

压缩预处理不会缩放图片。经过三次步长为 2 的下采样后，Decoder 输出尺寸为输入宽高向下取整到
8 的倍数。请使用宽高为 8 的倍数的图片，推荐使用模型训练时的 32×32 尺寸。

## 当前实现的边界

- 训练目标只有重建 MSE，没有联合优化码率与失真。
- 霍夫曼编码表是训练集统计得到的静态公共码表，不是学习式熵模型。
- `.bin` 不嵌入模型、整份码表或二者的哈希；必须由使用者保证模型与码表配套。
- 该二进制容器是本项目的教学格式，不是通用图片压缩标准。
- `evaluate.py` 给出经验熵理论估计；只有 `compress.py` 和 `evaluate_codec.py` 会生成真实位流。
- 模型主要面向 CIFAR-10 的 32×32 RGB 图片，并未针对高分辨率或任意分辨率图片训练。

这个项目适合用来理解神经网络图像压缩中的核心链路：

```text
表示学习 → 量化 → 符号概率 → 熵编码 → 容器格式 → 解码重建 → 码率/失真评估
```
