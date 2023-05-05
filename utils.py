import random, os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def CMI_loss(cls, z_y, z_r, s_batch):
    z_y_repeat = z_y.unsqueeze(1).expand(-1, z_y.shape[0], -1) # N x N (replica) x D
    z_r_repeat = z_r.unsqueeze(0).expand(z_y.shape[0], -1, -1) # N (replica) x N  x D

    z_yr = torch.cat([z_y_repeat, z_r_repeat], dim = -1).view(z_y.shape[0] **2, -1) # N^2 x 2D
    p_y = torch.sigmoid(cls(z_yr).view(z_y.shape[0], z_y.shape[0], -1)) # N x N x 1
    p_y_agg = p_y.mean(0)  # mean over different z_y
    
    H_y_cond_z = -(p_y_agg * torch.log(p_y_agg + 1e-7) + (1-p_y_agg) * torch.log(1-p_y_agg + 1e-7)).mean()
    
    s_idx = s_batch.view(-1) == 1
    p_a = s_idx.float().mean()

    p_ya1 = p_y[s_idx].mean(0) * torch.log(p_y[s_idx].mean(0) + 1e-7) + \
                (1-p_y[s_idx].mean(0)) * torch.log(1-p_y[s_idx].mean(0) + 1e-7)
    
    p_ya0 = p_y[~s_idx].mean(0) * torch.log(p_y[~s_idx].mean(0) + 1e-7) + \
                (1-p_y[~s_idx].mean(0)) * torch.log(1-p_y[~s_idx].mean(0) + 1e-7)
    
    H_y_cond_za = -(p_a * p_ya1 + (1 - p_a) * p_ya0).mean()
    
#     return H_y_cond_z - H_y_cond_za
    return H_y_cond_z, H_y_cond_za

def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv2d") != -1 or layer_name.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(layer.weight)
    elif layer_name.find("BatchNorm") != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_normal_(layer.weight)


def inv_lr_scheduler(optimizer, lr, iter, max_iter, gamma=10, power=0.75):
    learning_rate = lr * (1 + gamma * (float(iter) / float(max_iter))) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group["lr_mult"]
        i += 1

    return optimizer


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def aff(input):
    return torch.mm(input, torch.transpose(input, dim0=0, dim1=1))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
