import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np

class GPUAugmentation(nn.Module):
    def __init__(self, noise_prob=0.0, noise_types=None):
        super().__init__()
        self.prob = noise_prob
        # 如果 noise_types 为 None 或空，默认使用所有
        if not noise_types:
            self.noise_types = ["shift", "rotation", "color", "cutout", "erasing"]
        else:
            self.noise_types = noise_types
            
        print(f"✅ GPU Augmentation Initialized: P={self.prob}, Types={self.noise_types}")

        # 预定义一些 Transform (它们现在支持 Tensor 操作)
        # 旋转: -30 到 30 度
        self.rot_trans = T.RandomRotation(degrees=30)
        # 颜色抖动
        self.color_trans = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        # 平移 (Shift): 使用 RandomAffine 实现平移
        self.shift_trans = T.RandomAffine(degrees=0, translate=(0.2, 0.2))

    def forward(self, x):
        """
        x: Tensor (Batch_Size, 3, 32, 32), 已经在 GPU 上
        """
        # 如果概率为 0 或 训练模式关闭，直接返回
        if self.prob <= 0 or not self.training:
            return x

        # 为了保持 Batch 的多样性，我们生成一个掩码
        # 决定 Batch 中哪些图片需要加噪声
        B = x.shape[0]
        # 生成 B 个随机数，小于 prob 的位置为 True
        mask = torch.rand(B, device=x.device) < self.prob
        
        # 如果没有图片需要处理，直接返回
        if not mask.any():
            return x

        # 选出需要增强的图片
        # 注意：为了效率，我们只对选中的部分做操作，然后再放回去
        x_aug = x[mask].clone()

        # 依次应用选定的噪声
        for n_type in self.noise_types:
            if n_type == "rotation":
                x_aug = self.rot_trans(x_aug)
            
            elif n_type == "color":
                x_aug = self.color_trans(x_aug)
            
            elif n_type == "shift":
                x_aug = self.shift_trans(x_aug)
            
            elif n_type == "cutout":
                x_aug = self.batch_cutout(x_aug)
            
            elif n_type == "erasing":
                x_aug = self.batch_random_erasing(x_aug)

        # 将处理后的图片放回原 Batch
        x[mask] = x_aug
        return x

    def batch_cutout(self, img_batch, length=16):
        """
        在 GPU 上对 Batch 进行 Cutout
        """
        B, C, H, W = img_batch.shape
        mask = torch.ones((B, 1, H, W), device=img_batch.device)
        
        # 随机中心点
        y = torch.randint(0, H, (B,), device=img_batch.device)
        x = torch.randint(0, W, (B,), device=img_batch.device)

        y1 = torch.clamp(y - length // 2, 0, H)
        y2 = torch.clamp(y + length // 2, 0, H)
        x1 = torch.clamp(x - length // 2, 0, W)
        x2 = torch.clamp(x + length // 2, 0, W)

        # 这里的循环在 Python 层面做，但因为只是生成掩码，速度很快
        # 对 A100 来说，完全可以用 grid_sample 或 scatter 优化，但 for loop 更易读
        for i in range(B):
            mask[i, :, y1[i]:y2[i], x1[i]:x2[i]] = 0.
        
        return img_batch * mask

    def batch_random_erasing(self, img_batch, p=0.5, sl=0.02, sh=0.4, r1=0.3):
        """
        PyTorch 自带的 RandomErasing 通常只能处理单张，这里调用 torchvision 的 functional
        或者直接使用 T.RandomErasing (支持 batch)
        这里为了简单，我们直接实例化一个 T.RandomErasing 并在循环中调用
        """
        # torchvision 的 RandomErasing 现在支持 Tensor，但通常针对单张
        # 为了极速，我们简单实现一个随机噪声填充
        eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        # T.RandomErasing 需要 forward(img)，支持 (C,H,W) 或 (B,C,H,W)
        # 直接对整个 Batch 作用即可
        return eraser(img_batch)