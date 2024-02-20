import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_dataloader = DataLoader(mnist_train, batch_size=10, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=10, shuffle=True)
