import os

from torchvision import datasets

train_data = datasets.MNIST(root="./data", train=True, download=True)

os.mkdirs("test_images", exist_ok=True)

for i in range(10):
    image, label = train_data[i]
    image.save(f"test_images{os.sep}{i}.png")
