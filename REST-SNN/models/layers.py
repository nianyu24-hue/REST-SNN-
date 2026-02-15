import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act =I_LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act =I_LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.spike_counts = []
        self.recording = False
        self.layer_name = "unknown" 

    def forward(self, x, stage="train"):
        mem = torch.zeros_like(x[:, 0, ...])
        spike_pot = []
        T = x.shape[1]

        if self.recording:
            batch_spike_counts = []
            
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            if stage == "train":
                spike = self.act(mem - self.thresh, self.gama)
            elif stage == "infer":
                spike = (mem >= self.thresh).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
            
            if self.recording:
                avg_spike_rate = spike.mean().item()
                batch_spike_counts.append(avg_spike_rate)

        if self.recording:
            self.spike_counts.append(batch_spike_counts)
            
        return torch.stack(spike_pot, dim=1)
    
    def reset_stats(self):
        self.spike_counts = []
    
    def get_statistics(self):
        if not self.spike_counts:
            return None
            
        avg_rates = np.mean(self.spike_counts, axis=0)
        return {
            'time_steps': list(range(1, len(avg_rates)+1)),
            'rates': avg_rates.tolist(),
            'max': float(np.max(avg_rates)),
            'min': float(np.min(avg_rates)),
            'mean': float(np.mean(avg_rates)),
            'std': float(np.std(avg_rates))
        }
    

class AI_LIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, D):
  
        out = torch.round(input).clamp(0, D)  
        ctx.save_for_backward(input, out, torch.tensor([gama]), torch.tensor([D]))
        return out

    @staticmethod
    def backward(ctx, grad_output):


        (input, out, gama, D) = ctx.saved_tensors
        gama = gama.item()

        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None, None 


class AI_LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, D=2):
        super(I_LIFSpike, self).__init__()
        self.act = I_LIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.D = D

        self.spike_counts = []
        self.recording = False
        self.layer_name = "unknown"  #

    def forward(self, x, stage="train"):
        mem = torch.zeros_like(x[:, 0, ...])
        spike_pot = []
        T = x.shape[1]
        
        if self.recording:
            batch_spike_counts = []
        
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            if stage == "train":
                spike = self.act(mem - self.thresh, self.gama, self.D)
            elif stage == "infer":
                spike = (mem >= self.thresh).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)

            if self.recording:

                avg_spike_rate = spike.mean().item()
                batch_spike_counts.append(avg_spike_rate)
        if self.recording:
            self.spike_counts.append(batch_spike_counts)
            
        return torch.stack(spike_pot, dim=1)
    
    def reset_stats(self):
        self.spike_counts = []
    
    def plot_firing_rate(self, save_dir, epoch=None):
        if not self.spike_counts:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        avg_rates = np.mean(self.spike_counts, axis=0)
        time_steps = np.arange(1, len(avg_rates) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, avg_rates, 'o-', linewidth=2)
        plt.title(f'Average Firing Rate - {self.layer_name}')
        plt.xlabel('Time Step')
        plt.ylabel('Firing Rate')
        plt.grid(True)
        
        max_rate = np.max(avg_rates)
        min_rate = np.min(avg_rates)
        mean_rate = np.mean(avg_rates)
        plt.axhline(y=mean_rate, color='r', linestyle='--', alpha=0.7)
        plt.text(0.02, 0.95, f'Max: {max_rate:.4f}\nMin: {min_rate:.4f}\nMean: {mean_rate:.4f}', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        epoch_str = f'_epoch{epoch}' if epoch is not None else ''
        save_path = os.path.join(save_dir, f'firing_rate_{self.layer_name}{epoch_str}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def get_statistics(self):

        if not self.spike_counts:
            return None
            
        avg_rates = np.mean(self.spike_counts, axis=0)
        return {
            'time_steps': list(range(1, len(avg_rates)+1)),
            'rates': avg_rates.tolist(),
            'max': float(np.max(avg_rates)),
            'min': float(np.min(avg_rates)),
            'mean': float(np.mean(avg_rates)),
            'std': float(np.std(avg_rates))
        }

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x




class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)
        
        
        

    def forward(self, x):
        y = self.seqbn(x)
        return y