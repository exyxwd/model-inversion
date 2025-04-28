import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

def load_mnist_data():
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_set, test_set