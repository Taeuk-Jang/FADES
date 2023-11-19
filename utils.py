import random, os
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

        
def disloss(pred):
    return (-0.5 * torch.log(torch.sigmoid(pred) + 1e-5)).mean()

def permute_zs(zs):
    B, _ = zs[0].size()
    perm_z = []
    
    for z_i in zs:
        perm = torch.randperm(B).cuda()
        perm_z.append(z_i[perm])
    return torch.cat(perm_z, 1)

def CMI_loss(cls, z_y, z_r, s_batch, beta=0.5):
    z_y_repeat = z_y.unsqueeze(1).expand(-1, z_y.shape[0], -1) # N x N (replica) x D
    z_r_repeat = z_r.unsqueeze(0).expand(z_y.shape[0], -1, -1) # N (replica) x N  x D

    z_yr = torch.cat([z_y_repeat, z_r_repeat], dim = -1).view(z_y.shape[0] **2, -1) # N^2 x 2D
    p_y = torch.sigmoid(cls(z_yr).view(z_y.shape[0], z_y.shape[0], -1)) # N x N x 1
    p_y_agg = p_y.mean(0)  # mean over different z_y

    H_y_cond_z = -(p_y_agg * torch.log(p_y_agg + 1e-7) + (1-p_y_agg) * torch.log(1-p_y_agg + 1e-7)).mean()

    s_idx = s_batch.view(-1) == 1

    p_y_zra1 = p_y[s_idx, :][:, s_idx, :].mean(0)
    p_y_zra0 = p_y[~s_idx, :][:, ~s_idx, :].mean(0)

    H_y_cond_za = (p_y_zra1 * torch.log(p_y_zra1 + 1e-7) + \
        (1-p_y_zra1) * torch.log((1-p_y_zra1) + 1e-7)).sum() + \
    (p_y_zra0 * torch.log(p_y_zra0 + 1e-7) + \
        (1-p_y_zra0) * torch.log((1-p_y_zra0) + 1e-7)).sum()

    H_y_cond_za /= -s_batch.shape[0]

    I_y_az = (1-beta) * H_y_cond_z - H_y_cond_za
    
    return I_y_az

def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

