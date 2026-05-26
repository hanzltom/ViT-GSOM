import os
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(current_dir, "datasets")

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(root=datasets_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=datasets_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_set, test_set])

    return dataset

def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.FashionMNIST(root=datasets_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=datasets_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_set, test_set])

    return dataset

def load_usps():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.USPS(root=datasets_dir, train=True, download=True, transform=transform)
    test_set = datasets.USPS(root=datasets_dir, train=False, download=True, transform=transform)
    dataset = ConcatDataset([train_set, test_set])

    return dataset