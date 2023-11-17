import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import os
import shutil
import sys

import torch

# torch.manual_seed(2024)
# np.random.seed(2024)

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
parser.add_argument("--feat_dim", type=int, default=32)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--lambda_cls", type=float, default=1e3)
parser.add_argument("--lambda_tc", type=float, default=1e2)
parser.add_argument("--lambda_fair", type=float, default=1e2)
parser.add_argument("--log_name", type=str, default='')
parser.add_argument("--label", type=str, default='Smiling')
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

args.log_name += '{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}'.format(args.lambda_cls, args.lambda_tc, args.lambda_fair, args.alpha, args.beta)
# based on IntroVAE
dataname = f'{args.dataname}'

if args.dataname == 'celeba':
    dataname += f'_{args.label}'
    
save_dir = os.path.join('./save', dataname, args.log_name)

os.makedirs(save_dir, exist_ok = True)

# init log
with open(os.path.join(save_dir, "log.txt"), "a") as f:
    f.write(" ".join(sys.argv))
    
def write_loss_to_log(logging):
    with open(os.path.join(save_dir, "log.txt"), "a") as f:
        f.write(logging)

    
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
if args.dataname=='celeba':
    train_data = ImageDataset(train_set, args.dataname, 'Male', args.label, '/data/celebA/CelebA/Img/img_align_celeba', transform)

    valid_data = ImageDataset(valid_set, args.dataname, 'Male', args.label, '/data/celebA/CelebA/Img/img_align_celeba', transform_test)
    
elif args.dataname == 'UTK':
    train_data = ImageDataset(train_set, 'UTK', 1, 0, transform=transform)
    valid_data = ImageDataset(valid_set, 'UTK', 1, 0, transform=transform)
    test_data = ImageDataset(test_set, 'UTK', 1, 0, transform=transform)
    
elif args.dataname == 'dnc':
    train_data = ImageDataset(train_set, 'dnc', transform=transform)
    valid_data = ImageDataset(valid_set, 'dnc', transform=transform)
    test_data = ImageDataset(test_set, 'dnc', transform=transform)

epochs = args.epochs
samples = 10

lambda_cls = args.lambda_cls
lambda_cmi = args.lambda_fair

use_proj = args.proj
use_identity = True


hdim = 512
feat_dim = args.feat_dim
channel = [64, 128, 256, 512, 512]
encoder = nn.DataParallel(Encoder_FADES(hdim = hdim, feat_dim = feat_dim, channels = channel, image_size=img_size)).cuda()
decoder = nn.DataParallel(Decoder_Res(hdim = hdim, channels = channel, image_size=img_size)).cuda()
if use_proj:
    proj = nn.DataParallel(Projector_sep(dim_in = 4*feat_dim, dim_out = feat_dim)).cuda()
    
    
cls_y = nn.DataParallel(Classifier(input_dim = 2*feat_dim)).cuda()
cls_a = nn.DataParallel(Classifier(input_dim = 2*feat_dim)).cuda()
Dis = nn.DataParallel(Classifier(input_dim = 3*feat_dim)).cuda()

start_epoch = 0

if args.resume:
    encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pth'))['state_dict'])
    decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pth'))['state_dict'])
    cls_y.load_state_dict(torch.load(os.path.join(save_dir, 'cls_y.pth'))['state_dict'])
    cls_a.load_state_dict(torch.load(os.path.join(save_dir, 'cls_a.pth'))['state_dict'])
    Dis.load_state_dict(torch.load(os.path.join(save_dir, 'dis.pth'))['state_dict'])
    start_epoch = torch.load(os.path.join(save_dir, 'cls_a.pth'))['epoch']

vae_param_lst = [{'params':encoder.module.parameters()}, {'params':decoder.module.parameters(), 'lr':5e-4}]

cls_param_lst = cls_y.module.get_parameters() + cls_a.module.get_parameters()
optimizer_vae = torch.optim.Adam(vae_param_lst, lr = 1e-4, weight_decay = 5e-4)
optimizer_cls = torch.optim.Adam(cls_param_lst, lr = 1e-3, weight_decay = 1e-4)
optimizer_d = torch.optim.Adam(Dis.parameters(), lr = 1e-4, weight_decay = 1e-4)


criterion = torch.nn.BCEWithLogitsLoss()
criterion_ce = torch.nn.CrossEntropyLoss()

valid_iter = iter(validloader)
x_test, s_test, y_test = next(valid_iter)
x_test, s_test, y_test = x_test.cuda(), s_test.cuda().float().view(-1,1), y_test.cuda().float().view(-1,1)

for epoch in range(start_epoch, epochs + 1):
    if epoch > 15 and epoch <= 50:
        print('lr update')
        optimizer_vae = torch.optim.Adam(vae_param_lst, lr = 5e-5, weight_decay = 1e-5)
        optimizer_cls = torch.optim.Adam(cls_param_lst, lr = 5e-4, weight_decay = 1e-4)
        optimizer_d = torch.optim.Adam(Dis.parameters(), lr = 1e-5, weight_decay = 1e-5)
    elif epoch > 50:
        print('lr update')
        optimizer_vae = torch.optim.Adam(vae_param_lst, lr = 1e-5, weight_decay = 1e-5)
        optimizer_cls = torch.optim.Adam(cls_param_lst, lr = 1e-4, weight_decay = 1e-4)
        optimizer_d = torch.optim.Adam(Dis.parameters(), lr = 1e-5, weight_decay = 1e-5)
        
    step =0.
    loss_elbo_hist = 0.
    loss_recon_hist = 0.
    loss_prior_hist = 0.
    loss_fair_hist = 0.
    loss_y_hist = 0.
    loss_a_hist = 0.
    loss_tc_hist = 0.
    iters = tqdm(trainloader)
    
    encoder.train()
    decoder.train()
    if use_proj:
        proj.train()
    
    cls_y.train()
    cls_a.train()
    Dis.train()
    
    trainloader_2 = iter(DataLoader(train_data, batch_size=bs, shuffle=True, drop_last = True, num_workers = 32, pin_memory = True))
    
    for x_batch, s_batch, y_batch in iters:
        step += 1
        x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

        z, mu, logvar = encoder(x_batch)
        z_x, z_y, z_s, z_r = z
        mu_x, mu_y, mu_s, mu_r = mu
        logvar_x, logvar_y, logvar_s, logvar_r = logvar

        z = torch.cat(z, dim = -1)
        mu = torch.cat(mu, dim= -1)
        logvar = torch.cat(logvar, dim =-1)
            
        recon = decoder(z)

        ## ELBO loss
        loss_recon = F.l1_loss(recon, x_batch, reduction = 'sum')/z.shape[0]
        loss_prior = -0.5 * ((1 + logvar - mu ** 2 - logvar.exp())).sum()/z.shape[0]
        
        loss_elbo = loss_prior + loss_recon
        
        pred_y = cls_y(torch.cat([z_y, z_r], dim =-1))
        pred_a = cls_a(torch.cat([z_s, z_r], dim =-1))
        
        loss_CMI = 0.5 * (CMI_loss(cls_y, z_y, z_r, s_batch, args.beta) + CMI_loss(cls_a, z_s, z_r, y_batch, args.beta))
        
        loss_y = criterion(pred_y, y_batch)
        loss_a = criterion(pred_a, s_batch)
        cls_loss = loss_y + args.alpha * loss_a
        
        optimizer_vae.zero_grad()
        optimizer_cls.zero_grad()
        
        ## Discriminator 
        pred_d = torch.sigmoid(Dis(torch.cat([z_y,z_r,z_s], dim = -1)))
        loss_D = torch.log(pred_d/(1-pred_d + 1e-7) + 1e-7).mean()

        loss = loss_elbo + args.lambda_cls * cls_loss + args.lambda_tc * loss_D 
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
                
        loss.backward(retain_graph=True)
        optimizer_vae.step()
        optimizer_cls.step()
                
        loss_recon_hist += loss_recon.item()
        loss_prior_hist += loss_prior.item()
        loss_fair_hist += loss_CMI.item()
        loss_y_hist += loss_y.item()
        loss_a_hist += loss_a.item()
        loss_tc_hist += loss_D.item()
        
        ## Train Discriminator 

        x_batch_2, _, _ = next(trainloader_2)
        x_batch_2 = x_batch_2.cuda().float()
        z_2, _, _ = encoder(x_batch_2)
        
        z_x_2, z_y_2, z_s_2, z_r_2 = z_2

        pred_d_2_perm = permute_zs([z_y_2, z_r_2, z_s_2])
        pred_d_2 = torch.sigmoid(Dis(pred_d_2_perm))
        
        z, mu, logvar = encoder(x_batch)
        z_x, z_y, z_s, z_r = z
        pred_d = torch.sigmoid(Dis(torch.cat([z_y,z_r,z_s], dim = -1)))
        
        D_tc_loss = 0.5 * (-torch.log(pred_d + 1e-7).mean() \
                               - torch.log(1-pred_d_2 + 1e-7).mean())

        optimizer_d.zero_grad()
        D_tc_loss.backward()
        optimizer_d.step()

        
        iters.set_description('epoch : {}, Recon : {:.3f}, Prior : {:.3f}, CMI loss : {:.3f}, TC loss : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
                              .format(epoch, loss_recon_hist/step, loss_prior_hist/step, loss_fair_hist/step, loss_tc_hist/step, loss_y_hist/step, loss_a_hist/step))
        
    write_loss_to_log('\nepoch : {}, Recon : {:.3f}, Prior : {:.3f}, CMI loss : {:.3f}, TC loss : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
                              .format(epoch, loss_recon_hist/step, loss_prior_hist/step, loss_fair_hist/step, loss_tc_hist/step, loss_y_hist/step, loss_a_hist/step))
        
    if epoch % 10 == 0:
        encoder.eval()
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
                
                z, _, _ = encoder(x_test)
                z_x, z_y, z_s, z_r = z
                z = torch.cat(z, dim = -1)
                
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
                plt.savefig(os.path.join(save_dir, 'recon_{}-{}.pdf'.format(epoch, sample_idx)), bbox_inches = 'tight')

    torch.save({'state_dict':encoder.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'encoder.pth'))
    torch.save({'state_dict':decoder.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'decoder.pth'))
    torch.save({'state_dict':cls_y.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'cls_y.pth'))
    torch.save({'state_dict':cls_a.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'cls_a.pth'))
    torch.save({'state_dict':Dis.state_dict(), 'epoch':epoch}, os.path.join(save_dir, 'dis.pth'))

