import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def build_cifar(use_cifar10=True, download=True, noise_prob=0.0, noise_types=None):
    # 注意：这里的 noise_prob 和 noise_types 参数虽然传进来了
    # 但是我们在 data_loader 这一层 **不再使用它们**
    # 所有的噪声都留给 main.py 里的 GPU 模块处理

    # 训练集基础增强 (保留 Crop 和 Flip 是为了不破坏原图尺寸和基础泛化)
    aug = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # 转为 Tensor
    ]

    # 归一化参数
    if use_cifar10:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_cls = CIFAR10
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        dataset_cls = CIFAR100

    # 归一化保留在 CPU 做
    aug.append(transforms.Normalize(mean, std))
    
    transform_train = transforms.Compose(aug)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = dataset_cls(root='./data', train=True, download=download, transform=transform_train)
    val_dataset = dataset_cls(root='./data', train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset