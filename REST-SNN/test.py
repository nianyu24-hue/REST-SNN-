import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from models.MS_ResNet import *
import data_loaders
from functions import seed_all
from observation_noise import apply_all_noise 

# 显卡配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Robust-SNN Specific Noise Testing')
parser.add_argument('--model_path', default='checkpoint_best.pth', type=str, help='权重路径')
parser.add_argument('--noise_types', nargs='+', default=["shift", "rotation", "color", "cutout", "erasing"], 
                    help='要测试的噪声组合')
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('-T', '--time', default=6, type=int)
parser.add_argument('--seed', default=1000, type=int)

args = parser.parse_args()

@torch.no_grad()
def test_with_custom_noise(model, test_loader, device, noise_types=None):
    model.eval()
    
    # 1. 初始化统计变量
    correct = 0
    total = 0
    
    # 2. 预设归一化参数 (CIFAR-100 标准值)
    # 保持在 CPU 上用于处理 Numpy 数据，或稍后移至 GPU
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
    
    # 判断是否为 Clean 模式
    is_clean = (noise_types is None or len(noise_types) == 0 or 
                noise_types[0].lower() == "none" or noise_types[0] == "")
    
    if is_clean:
        print(">>> 确认：正在执行 Clean Accuracy 测试，跳过所有图像扰动。")
    else:
        print(f">>> 正在应用噪声类型: {noise_types}")

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        
        # --- 步骤 A: 噪声处理 (仅在非 Clean 模式下) ---
        if not is_clean:
            # 数据增强通常在 CPU 上使用 PIL/Numpy 运行
            # a. 撤销归一化 [0, 1]
            inputs = inputs * std + mean 
            
            # b. 转换为 Numpy 进行噪声注入
            inputs_np = inputs.numpy().transpose(0, 2, 3, 1)
            noisy_inputs = []
            for i in range(inputs_np.shape[0]):
                # 确保像素在 [0, 1] 范围内并转为 uint8
                img_uint8 = (np.clip(inputs_np[i], 0, 1) * 255).astype(np.uint8)
                # 调用自定义噪声函数
                processed_img = apply_all_noise(img_uint8, select_types=noise_types)
                noisy_inputs.append(processed_img.astype(np.float32) / 255.0)
            
            # c. 转回 Tensor 并重新归一化
            inputs = torch.from_numpy(np.array(noisy_inputs)).permute(0, 3, 1, 2)
            inputs = (inputs - mean) / std

        # --- 步骤 B: 设备对齐 (核心修复) ---
        # 必须确保 inputs 和 targets 都在模型所在的 GPU 上 [cite: 339]
        inputs = inputs.to(device)
        targets = targets.to(device)

        # --- 步骤 C: 模型前向传播 ---
        model_out = model(inputs)
        # GAC 结构通常返回 (脉冲输出, 注意力掩码) [cite: 1, 419]
        if isinstance(model_out, tuple):
            outputs, _ = model_out
        else:
            outputs = model_out
            
        # SNN 计算 $T$ 个时间步的平均发放率 (Rate Coding) [cite: 72, 166, 178]
        mean_out = outputs.mean(1) 
        _, predicted = mean_out.max(1)
        
        # --- 步骤 D: 结果统计 ---
        total += targets.size(0)
        # 此时 predicted 和 targets 都在 device 上，不会触发 RuntimeError
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print(f'Progress: {batch_idx}/{len(test_loader)} | Acc: {100.*correct/total:.2f}%')

    return 100. * correct / total

if __name__ == '__main__':
    seed_all(args.seed)

    # 加载数据集
    _, val_dataset = data_loaders.build_cifar(use_cifar10=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化 (msresnet18 针对 CIFAR-100)
    parallel_model = msresnet18(num_classes=100)
    # SNN 仿真时间步 T [cite: 183]
    parallel_model.T = args.time 
    parallel_model = torch.nn.DataParallel(parallel_model)
    parallel_model.to(device)

    # 加载权重 [cite: 32]
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        parallel_model.module.load_state_dict(state_dict, strict=False)
        print(f"Loaded weight from {args.model_path}")
    else:
        print(f"Warning: Model path {args.model_path} not found!")

    # 执行测试并打印
    acc = test_with_custom_noise(parallel_model, test_loader, device, noise_types=args.noise_types)
    
    print("\n" + "="*35)
    print(f"最终测试准确率: {acc:.3f}%")
    print(f"使用的噪声集: {args.noise_types}")
    print("="*35)