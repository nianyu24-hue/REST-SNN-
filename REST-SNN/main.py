import argparse
import os
import time
import datetime
import csv
import torch.nn.parallel
import torch.optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

from models.MS_ResNet import *
from functions import seed_all, get_logger
from data_loaders import build_cifar


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Gated Attention Coding')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, dest='lr')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('-T', '--time', default=6, type=int, help='snn simulation time steps')
parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=250, type=int)

parser.add_argument('--cutmix_beta', default=1.0, type=float)
parser.add_argument('--cutmix_prob', default=0.5, type=float)

parser.add_argument('--alpha', default=1.0, type=float, help='GMS 权重')
parser.add_argument('--beta', default=0.05, type=float, help='TCS 权重')


parser.add_argument('--noise_prob', type=float, default=0.0)
parser.add_argument('--noise_types', nargs='+', default=None)
parser.add_argument('--wandb_project', type=str, default='CIFAR100-GAC')

args = parser.parse_args()


CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def calc_gms_loss_normalized(images, masks):
    if not images.requires_grad:
        images.requires_grad_(True)
    mask_sum = masks.sum()
    grad_mask = torch.autograd.grad(outputs=mask_sum, inputs=images, 
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    if grad_mask is None:
        return torch.tensor(0.0, device=images.device)
    grad_flat = grad_mask.reshape(grad_mask.shape[0], -1)
    grad_flat = torch.nan_to_num(grad_flat, nan=0.0)
    # L2 Norm Squared Mean
    return grad_flat.pow(2).mean()

def calc_tcs_loss(masks):
    if masks.dim() < 2 or masks.shape[1] <= 1:
        return torch.tensor(0.0, device=masks.device)
    diff = masks[:, 1:] - masks[:, :-1]
    return diff.pow(2).mean()


def train(model, device, train_loader, criterion, optimizer, epoch, args, gpu_augmenter=None):
    model.train()
    if gpu_augmenter:
        gpu_augmenter.train()
    
    metrics = {'loss': 0, 'gms': 0, 'tcs': 0, 'acc1': 0, 'acc5': 0, 'total': 0}
    loop = tqdm(train_loader, total=len(train_loader), desc=f'Train Ep {epoch}', ncols=110)

    for images, labels in loop:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)

        if gpu_augmenter is not None:
            with torch.no_grad():

                images = images * CIFAR100_STD + CIFAR100_MEAN
                images = torch.clamp(images, 0, 1)

                images = gpu_augmenter(images)

                images = (images - CIFAR100_MEAN) / CIFAR100_STD
        
        # 2. CutMix
        r = np.random.rand(1)
        use_cutmix = False
        if args.cutmix_beta > 0 and r < args.cutmix_prob:
            use_cutmix = True
            lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
            rand_index = torch.randperm(images.size()[0]).to(device)
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        if args.alpha > 0:
            images.requires_grad_(True)
        
        model_out = model(images)
        if isinstance(model_out, tuple):
            outputs, masks = model_out
        else:
            outputs, masks = model_out, None

        mean_out = outputs.mean(1)

        if use_cutmix:
            loss_task = criterion(mean_out, target_a) * lam + criterion(mean_out, target_b) * (1. - lam)
        else:
            loss_task = criterion(mean_out, labels)
        
        loss_gms = calc_gms_loss_normalized(images, masks) if (masks is not None and args.alpha > 0) else torch.tensor(0.0, device=device)
        loss_tcs = calc_tcs_loss(masks) if (masks is not None and args.beta > 0) else torch.tensor(0.0, device=device)
        
        loss = loss_task + (args.alpha * loss_gms) + (args.beta * loss_tcs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = labels.size(0)
        metrics['total'] += bs
        metrics['loss'] += loss.item() * bs
        metrics['gms'] += loss_gms.item() * bs
        metrics['tcs'] += loss_tcs.item() * bs
        
        acc1, acc5 = accuracy(mean_out, labels, topk=(1, 5))
        metrics['acc1'] += acc1.item() * bs / 100.0
        metrics['acc5'] += acc5.item() * bs / 100.0
        
        loop.set_postfix(loss=loss.item(), top1=acc1.item(), gms=loss_gms.item())

    return (metrics['loss']/metrics['total'], metrics['acc1']/metrics['total']*100, 
            metrics['acc5']/metrics['total']*100, metrics['gms']/metrics['total'], 
            metrics['tcs']/metrics['total'])


@torch.no_grad()
def test(model, test_loader, device, gpu_augmenter=None, apply_noise=False):
    model.eval()
    

    if apply_noise and gpu_augmenter:
        gpu_augmenter.train()
    elif gpu_augmenter:
        gpu_augmenter.eval()

    total = 0.0
    correct_top1 = 0.0
    correct_top5 = 0.0
    all_targets = []
    all_preds = []

    desc = 'Test (Noisy)' if apply_noise else 'Test (Clean)'
    loop = tqdm(test_loader, desc=desc, ncols=100)

    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        

        if apply_noise and gpu_augmenter:
            inputs = inputs * CIFAR100_STD + CIFAR100_MEAN
            inputs = torch.clamp(inputs, 0, 1)
            inputs = gpu_augmenter(inputs)
            inputs = (inputs - CIFAR100_MEAN) / CIFAR100_STD
            
        model_out = model(inputs)
        if isinstance(model_out, tuple):
            outputs, _ = model_out
        else:
            outputs = model_out
            
        mean_out = outputs.mean(1)
        acc1, acc5 = accuracy(mean_out, targets, topk=(1, 5))
        
        batch_size = targets.size(0)
        total += batch_size
        correct_top1 += acc1.item() * batch_size / 100.0
        correct_top5 += acc5.item() * batch_size / 100.0

        _, predicted = mean_out.max(1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        loop.set_postfix(acc=acc1.item())

    final_acc1 = correct_top1 / total * 100.0
    final_acc5 = correct_top5 / total * 100.0
    
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return final_acc1, final_acc5, precision, recall, f1

if __name__ == '__main__':
    seed_all(args.seed)

    BASE_OUTPUT_DIR = '####'
    current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.noise_prob == 0.0:
        types_str = "None"
    elif args.noise_types is None:
        types_str = "All"
    else:
        types_str = "-".join(args.noise_types)

    folder_name = f"Noise_{args.noise_prob}_{types_str}_a{args.alpha}_b{args.beta}_{current_time_str}"
    save_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created output directory: {save_dir}")

    csv_file_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'LR', 'Train_Loss', 'GMS_Loss', 'TCS_Loss', 
                         'Train_Acc1', 'Train_Acc5', 
                         'Val_Clean_Acc1', 'Val_Clean_Acc5', 'Val_Noisy_Acc1', 'Val_Noisy_Acc5'])

    wandb.init(project=args.wandb_project, name=folder_name, config=args, dir=save_dir)
    log_file_path = os.path.join(save_dir, 'training.log')
    logger = get_logger(log_file_path)
    logger.info(f'Output Directory: {save_dir}')
    logger.info(f'Regularization: Alpha={args.alpha}, Beta={args.beta}')

    try:
        from gpu_noise import GPUAugmentation
        gpu_augmenter = GPUAugmentation(noise_prob=args.noise_prob, noise_types=args.noise_types).to(device)
    except ImportError:
        print("Warning: gpu_noise not found.")
        gpu_augmenter = None

    use_cifar10 = False 
    train_dataset, val_dataset = build_cifar(use_cifar10=use_cifar10, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    parallel_model = msresnet18(num_classes=100)
    parallel_model.T = args.time
    parallel_model = torch.nn.DataParallel(parallel_model).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(parallel_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    top_k_models = []
    MAX_KEEP = 5

    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5, epoch_gms, epoch_tcs = train(
            parallel_model, device, train_loader, criterion, optimizer, epoch, args, gpu_augmenter
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        val_c_acc1, val_c_acc5, _, _, val_c_f1 = test(
            parallel_model, test_loader, device, apply_noise=False
        )

        val_n_acc1, val_n_acc5, _, _, val_n_f1 = test(
            parallel_model, test_loader, device, gpu_augmenter=gpu_augmenter, apply_noise=True
        )

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/gms_loss": epoch_gms,
            "train/tcs_loss": epoch_tcs,
            "train/top1_acc": train_acc1,
            "lr": current_lr,
            
            "val/clean_acc1": val_c_acc1,
            "val/clean_acc5": val_c_acc5,
            
            "val/noisy_acc1": val_n_acc1,
            "val/noisy_acc5": val_n_acc5,
        })
        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, current_lr, train_loss, epoch_gms, epoch_tcs,
                             train_acc1, train_acc5, 
                             val_c_acc1, val_c_acc5, val_n_acc1, val_n_acc5])

        logger.info(f'Ep {epoch}: Clean={val_c_acc1:.2f}% | Noisy={val_n_acc1:.2f}%')

        target_acc = val_n_acc1 
        
        if len(top_k_models) < MAX_KEEP or target_acc > top_k_models[-1]['acc']:
            ckpt_name = f'checkpoint_ep{epoch}_noisy{target_acc:.2f}.pth'
            ckpt_path = os.path.join(save_dir, ckpt_name)
            
            torch.save(parallel_model.module.state_dict(), ckpt_path)
            logger.info(f'Saving top robust model: {ckpt_name}')
            
            top_k_models.append({'acc': target_acc, 'epoch': epoch, 'path': ckpt_path})
            top_k_models.sort(key=lambda x: x['acc'], reverse=True)
            
            if len(top_k_models) > MAX_KEEP:
                worst_model = top_k_models.pop()
                try:
                    if os.path.exists(worst_model['path']):
                        os.remove(worst_model['path'])
                except: pass

    logger.info(f'Training Finished.')
    wandb.finish()