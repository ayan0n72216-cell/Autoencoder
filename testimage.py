from torchvision.datasets import CIFAR10
from torchvision import transforms

test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
)

image, label = test_dataset[0]
image.save("test.png")

tensor = transforms.ToTensor()(image)

print("图片模式：", image.mode)       # RGB
print("图片尺寸：", image.size)       # (32, 32)
print("张量形状：", tensor.shape)     # [3, 32, 32]
print("类别编号：", label)