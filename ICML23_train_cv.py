import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import os
import shutil
import sys

import torch
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision

from dataloader import mnist_usps, mnist_reverse, FaceLandmarksDataset, ImageLoader, ImageDataset
from module import *
from utils import *
import argparse
from PIL import Image

# from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--lambda_cls", type=float, default=1e3)
parser.add_argument("--lambda_fair", type=float, default=1e2)
parser.add_argument("--loss_opt", type=str, default='CMI')
parser.add_argument("--log_name", type=str, default='celeba')
parser.add_argument("--shared", type=bool, default=True)
parser.add_argument("--proj", type=bool, default=False)
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()


# based on IntroVAE
save_dir = os.path.join('./save', args.log_name)
fig_dir = os.path.join('./figure', args.log_name)

print('shared : ', args.shared)

os.makedirs(save_dir, exist_ok = True)
os.makedirs(fig_dir, exist_ok = True)
    
filename = sys.argv[0]
shutil.copyfile(filename, os.path.join(save_dir, filename))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_size = 128
crop_size = 128

orig_w = 178
orig_h = 218
orig_min_dim = min(orig_w, orig_h)

transform = transforms.Compose([
    transforms.CenterCrop(orig_min_dim),
    transforms.Resize(img_size),
    transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.CenterCrop(orig_min_dim),
    transforms.Resize(img_size),
    transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set, valid_set, test_set = ImageLoader()
bs = args.bs
bs_val = 10
#Smiling or Attractive
train_data = ImageDataset(train_set, 'celeba', 'Male', 'Smiling', '/data/celebA/CelebA/Img/img_align_celeba', transform)
trainloader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last = True, num_workers = 32, pin_memory = True)

valid_data = ImageDataset(valid_set, 'celeba', 'Male', 'Smiling', '/data/celebA/CelebA/Img/img_align_celeba', transform_test)
validloader = DataLoader(valid_data, batch_size=bs_val, shuffle=False, num_workers = 32, pin_memory = True)

epochs = args.epochs
samples = 10

lambda_cls = args.lambda_cls
lambda_cmi = args.lambda_fair

use_proj = args.proj
shared = args.shared
use_identity = True


hdim = 512
feat_dim = 32
channel = [64, 128, 256, 512, 512]
if shared:
    encoder = nn.DataParallel(Encoder_FADES(hdim = hdim, feat_dim = feat_dim, channels = channel, image_size=img_size)).cuda()
else:
    encoder_x = nn.DataParallel(Encoder_Res(hdim = 1024-3*feat_dim)).cuda()
    encoder_y = nn.DataParallel(Encoder_Res(hdim = feat_dim)).cuda()
    encoder_s = nn.DataParallel(Encoder_Res(hdim = feat_dim)).cuda()
    encoder_r = nn.DataParallel(Encoder_Res(hdim = feat_dim)).cuda()
decoder = nn.DataParallel(Decoder_Res(hdim = hdim, channels = channel, image_size=img_size)).cuda()
if use_proj:
    proj = nn.DataParallel(Projector_sep(dim_in = 4*feat_dim, dim_out = feat_dim)).cuda()
    
    
cls_y = nn.DataParallel(Classifier(input_dim = 2*feat_dim)).cuda()
cls_a = nn.DataParallel(Classifier(input_dim = 2*feat_dim)).cuda()
start_epoch = 0

if args.resume:
    encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pth'))['state_dict'])
    decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pth'))['state_dict'])
    cls_y.load_state_dict(torch.load(os.path.join(save_dir, 'cls_y.pth'))['state_dict'])
    cls_a.load_state_dict(torch.load(os.path.join(save_dir, 'cls_a.pth'))['state_dict'])
    start_epoch = torch.load(os.path.join(save_dir, 'cls_a.pth'))['epoch']

if shared:
    vae_param_lst = [{'params':encoder.module.parameters()}, {'params':decoder.module.parameters(), 'lr':1e-3}]
else:
    vae_param_lst = list(encoder_x.module.parameters()) +list(encoder_s.module.parameters()) + list(encoder_y.module.parameters()) \
                +list(encoder_r.module.parameters()) + list(decoder.module.parameters()) 
if use_proj:
    vae_param_lst += list(proj.module.parameters())
cls_param_lst = cls_y.module.get_parameters() + cls_a.module.get_parameters()

optimizer_vae = torch.optim.Adam(vae_param_lst, lr = 1e-4, weight_decay = 1e-4)
optimizer_cls = torch.optim.Adam(cls_param_lst, lr = 1e-3, weight_decay = 1e-3)

criterion = torch.nn.BCEWithLogitsLoss()
criterion_ce = torch.nn.CrossEntropyLoss()

valid_iter = iter(validloader)
x_test, s_test, y_test = valid_iter.next()
x_test, s_test, y_test = x_test.cuda(), s_test.cuda().float().view(-1,1), y_test.cuda().float().view(-1,1)

for epoch in range(start_epoch, epochs + 1):
    step =0.
    loss_elbo_hist = 0.
    loss_recon_hist = 0.
    loss_prior_hist = 0.
    loss_fair_hist = 0.
    loss_y_hist = 0.
    loss_a_hist = 0.
    iters = tqdm(trainloader)
    
    if shared:
        encoder.train()
    else:
        encoder_x.train()
        encoder_y.train()
        encoder_s.train()
        encoder_r.train()
    decoder.train()
    if use_proj:
        proj.train()
    
    cls_y.train()
    cls_a.train()
    
    for x_batch, s_batch, y_batch in iters:
        step += 1
        x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

        if shared:
            z, mu, logvar = encoder(x_batch)
            z_x, z_y, z_s, z_r = z
            mu_x, mu_y, mu_s, mu_r = mu
            logvar_x, logvar_y, logvar_s, logvar_r = logvar
            
            z = torch.cat(z, dim = -1)
            mu = torch.cat(mu, dim= -1)
            logvar = torch.cat(logvar, dim =-1)
            
        else:
            z_x, mu_x, logvar_x = encoder_x(x_batch)
            z_y, mu_y, logvar_y = encoder_y(x_batch)
            z_s, mu_s, logvar_s = encoder_s(x_batch)
            z_r, mu_r, logvar_r = encoder_r(x_batch)
        
            mu = torch.cat([mu_x, mu_y, mu_s, mu_r], dim = 1)
            logvar = torch.cat([logvar_x, logvar_y, logvar_s, logvar_r], dim = 1)
        
        recon = decoder(z)

        loss_recon = F.l1_loss(recon, x_batch, reduction = 'sum')
        loss_prior = -0.5 * ((1 + logvar - mu ** 2 - logvar.exp())).sum()
        loss_recon /= z.shape[0]
        loss_prior /= z.shape[0]
        
        loss_elbo = loss_prior + loss_recon
        
        pred_y = cls_y(torch.cat([z_y, z_r], dim =-1))
        pred_a = cls_a(torch.cat([z_s, z_r], dim =-1))
        
        if args.loss_opt == 'CMI':
            H_y_cond_z_1, H_y_cond_za = CMI_loss(cls_y, z_y, z_r, (pred_a>=0).int().detach())
            H_y_cond_z_2, H_y_cond_zs = CMI_loss(cls_a, z_s, z_r, (pred_y>=0).int().detach())
            loss_CMI = 0.5 * (H_y_cond_z_1 + H_y_cond_z_2 - H_y_cond_za - H_y_cond_zs)
        
        loss_y = criterion(pred_y, y_batch)
        loss_a = criterion(pred_a, s_batch)
        cls_loss = loss_y + loss_a
        
        optimizer_vae.zero_grad()
        optimizer_cls.zero_grad()
        loss = loss_elbo + args.lambda_cls * cls_loss
        loss_fair = args.lambda_fair * loss_CMI
        if epoch > 1:
            for params in cls_y.module.parameters():
                params.require_grad = False
            for params in cls_a.module.parameters():
                params.require_grad = False
                
            loss_fair.backward(retain_graph = True)

            for params in cls_y.module.parameters():
                params.require_grad = True
            for params in cls_a.module.parameters():
                params.require_grad = True
                
        loss.backward(retain_graph = True)
        optimizer_vae.step()
        optimizer_cls.step()
                
        loss_recon_hist += loss_recon.item()
        loss_prior_hist += loss_prior.item()
        loss_fair_hist += loss_fair.item()
        loss_y_hist += loss_y.item()
        loss_a_hist += loss_a.item()
        
        iters.set_description('epoch : {}, Recon : {:.3f}, Prior : {:.3f}, CMI loss : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
                              .format(epoch, loss_recon_hist/step, loss_prior_hist/step, loss_fair_hist/step, loss_y_hist/step, loss_a_hist/step))
        
        
    if epoch % 10 == 0:
        if shared:
            encoder.eval()
        else:
            encoder_x.eval()
            encoder_i.eval()
            encoder_r.eval()
        decoder.eval()
        
        if use_proj:
            proj.eval()

        cls_y.eval()
        cls_a.eval()
    
        with torch.no_grad():
            pred_a = torch.sigmoid(pred_a)
            pred_a[pred_a>=0.5] = 1
            pred_a[pred_a<0.5] = 0
            pred_y = torch.sigmoid(pred_y)
            pred_y[pred_y>=0.5] = 1
            pred_y[pred_y<0.5] = 0
            
            acc_a = (pred_a == s_batch).float().mean()
            acc_y = (pred_y == y_batch).float().mean()

            print('epoch : {}, ELBO : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
                              .format(epoch, loss_elbo.item(), loss_y.item(),loss_a.item()))
            print('epoch : {}, Acc Y : {:.3f}, Acc A : {:.3f}'.format(epoch, acc_y, acc_a))

            for sample_idx in [0,1]:
                label_lst, img_lst = [],[]
                
                if shared:
                    z, _, _ = encoder(x_test)
                    z_x, z_y, z_s, z_r = z
                    z = torch.cat(z, dim = -1)
                else:
                    z_x, _, _ = encoder_x(x_test)
                    z_y, _, _ = encoder_y(x_test)
                    z_s, _, _ = encoder_s(x_test)
                    z_r, _, _ = encoder_r(x_test)

                img_lst.append( torchvision.utils.make_grid(x_test[: samples], nrow=samples).cpu().permute(1,2,0) )
                label_lst.append('Original')
                
                z_ent = torch.cat([z_y, z_s, z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append('$Recon$')

                z_ent = torch.cat([z_y, z_s, z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append('$[z_x^{' + "({})".format(sample_idx)+'}, z_y, z_s, z_r]$')

                z_ent = torch.cat([z_y[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_s, z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '$[z_x, z_y^{' + "({})".format(sample_idx)+'}, z_s, z_r]$')

                z_ent = torch.cat([z_y, z_s[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '$[z_x, z_y, z_s^{' + "({})".format(sample_idx)+'}, z_r]$')
                
                z_ent = torch.cat([z_y, z_s, z_r[sample_idx].unsqueeze(0).repeat(bs_val, 1)], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '$[z_x, z_y, z_s, z_r^{' + "({})".format(sample_idx)+'}]$')

                z_ent = torch.cat([z_y, z_s[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_r[sample_idx].unsqueeze(0).repeat(bs_val, 1)], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '$[z_x, z_y, z_s^{' + "({})".format(sample_idx)+'}, z_r^{' + "({})".format(sample_idx)+'}]$')
                
                z_ent = torch.cat([z_y, z_s, z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent[sample_idx].unsqueeze(0).repeat(bs_val, 1)], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '$[z_x, z_y^{' + "({})".format(sample_idx)+'}, z_s^{' + "({})".format(sample_idx)+'}, z_r^{' + "({})".format(sample_idx)+'}]$')

                z_ent = torch.cat([z_y, z_s, z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([torch.randn_like(z_x).cuda(), z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '[$N, z_y, z_s, z_r]$')
                
                z_ent = torch.cat([torch.randn_like(z_y).cuda(), z_s, z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([torch.randn_like(z_x).cuda(), z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '[$z_x, N, z_s, z_r]$')
                
                z_ent = torch.cat([z_y, torch.randn_like(z_s).cuda(), z_r], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([torch.randn_like(z_x).cuda(), z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '[$z_x, z_y, N, z_r]$')
                
                z_ent = torch.cat([z_y, z_s, torch.randn_like(z_r).cuda()], dim = 1)
                if use_proj:
                    z_ent = proj(z_ent)
                z = torch.cat([z_x, z_ent], dim = 1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0))
                label_lst.append( '[$z_x, z_y, z_s, N]$')
                
                
                #img_lst = [recon_1, recon_2, recon_3, recon_4, recon_5, recon_6, recon_7, recon_8, image_1]
                img = torch.cat(img_lst, 0).detach().numpy()

                img_h = x_test.shape[2]
                img_w = int(x_test.shape[3])

                label_list = []
                for i in range(samples):
                    tick = ''
                    if y_test[i] == 1:
                        tick += 'Smiling, '
                    else:
                        tick += 'Not Smiling, '
                    if s_test[i] == 1:
                        tick += 'Male'
                    else:
                        tick += 'Female'
                    label_list.append(tick)

                plt.figure(figsize = (24,24))
                plt.imshow(img)
                plt.xticks([img_w*0.5 + i * img_w for i in range(samples)], \
                          label_list)
                plt.yticks([img_h*0.5 + i * img_h for i in range(len(img_lst))], label_lst, fontsize = 17)
                plt.savefig(os.path.join(fig_dir, 'recon_{}-{}.pdf'.format(epoch, sample_idx)), bbox_inches = 'tight')
               
    if shared:
        torch.save({'state_dict':encoder.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'encoder.pth'))
    else:
        torch.save({'state_dict':encoder_x.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'encoder_x.pth'))
        torch.save({'state_dict':encoder_y.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'encoder_y.pth'))
        torch.save({'state_dict':encoder_s.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'encoder_s.pth'))
        torch.save({'state_dict':encoder_r.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'encoder_r.pth'))
    
    torch.save({'state_dict':decoder.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'decoder.pth'))
    if use_proj:
        torch.save({'state_dict':proj.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'proj.pth'))
    torch.save({'state_dict':cls_y.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'cls_y.pth'))
    torch.save({'state_dict':cls_a.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'cls_a.pth'))

