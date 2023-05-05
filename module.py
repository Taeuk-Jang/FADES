import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from sklearn.cluster import KMeans
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from utils import init_weights
from scipy.optimize import linear_sum_assignment
import torchvision.models as models

import torch.nn.init as init

class Encoder_tab_orth(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Encoder_tab_orth, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        
        self.linear3_1 = nn.Linear(64, latent_dim)
        self.linear3_2 = nn.Linear(64, latent_dim)

        self.relu = nn.ReLU()
        
        for m in self.children():
            weights_init_kaiming(m)

    def reparameterize(self, mean_t, mean_s, log_var_t, log_var_s):
        if self.training:
            z1 = mean_t + torch.exp(log_var_t/2) @ torch.normal(torch.from_numpy(np.array([0.,1.,0.]).T).float(), torch.eye(3)).cuda()
            z2 = mean_s + torch.exp(log_var_s/2) @ torch.normal(torch.from_numpy(np.array([1.,0.,1.]).T).float(), torch.eye(3)).cuda()
            return z1, z2
        else:
            return mean_t, mean_s

    def forward(self, x):
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        
        mu, logvar = self.linear3_1(x), self.linear3_2(x)
        
        mean_t, mean_s = mu.split(int(mu.shape[1]/2), dim = 1)
        log_var_t, log_var_s = logvar.split(int(logvar.shape[1]/2), dim = 1)
        z1, z2 = self.reparameterize(mean_t, mean_s, log_var_t, log_var_s)
        return z1, z2, mean_t, mean_s, log_var_t, log_var_s
    
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        
        
class Encoder_tab(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Encoder_tab, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        
        self.linear3_1 = nn.Linear(64, latent_dim)
        self.linear3_2 = nn.Linear(64, latent_dim)

        self.relu = nn.LeakyReLU(0.2)
#         self.relu = nn.ReLU()
        
        for m in self.children():
            weights_init_kaiming(m)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        
        mu, logvar = self.linear3_1(x), self.linear3_2(x)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder_tab(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Decoder_tab, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, input_dim)

        self.relu = nn.LeakyReLU(0.2)
#         self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        return x
    
    
class MLP(nn.Module):
    def __init__(self, input_dim = 32, hidden_dim = 128):
        super(MLP, self).__init__()
        self.input_dim = input_dim 
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    
class Classifier(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 256, output_dim = 1):
        super(Classifier, self).__init__()
        self.input_dim = input_dim 
        self.dense1 = nn.Linear(input_dim, hidden_dim)
#         self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        for m in self.children():
            weights_init_kaiming(m)
        
    def forward(self, x):
        x = self.leakyrelu(self.dense1(x))
#         x = self.leakyrelu(self.dense2(x))
        x = self.dense3(x)
        
        return x
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

class Classifier_simple(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim = 256, output_dim = 1):
        super(Classifier_simple, self).__init__()
        self.input_dim = input_dim 
#         self.dense1 = nn.Linear(input_dim, hidden_dim)
#         self.dense2 = nn.Linear(hidden_dim, hidden_dim)
#         self.dense3 = nn.Linear(hidden_dim, output_dim)
        self.dense1 = nn.Linear(input_dim, output_dim)
#         self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
#         x = self.leakyrelu(self.dense1(x))
#         x = self.leakyrelu(self.dense2(x))
#         x = self.dense3(x)
        x = self.dense1(x)
        return x
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]
    
class Encoder_orth(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder_orth, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, 2*hdim)           
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        mean_x, mean_t, mean_s = mu[:, :-64], mu[:, -64:-32], mu[:, -32:]
        log_var_x, log_var_t, log_var_s = logvar[:, :-64], logvar[:, -64:-32], logvar[:, -32:]
        
        z, z1, z2 = self.reparameterize(mean_x, mean_t, mean_s, log_var_x, log_var_t, log_var_s)
        return z, z1, z2, mean_x, mean_t, mean_s, log_var_x, log_var_t, log_var_s
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mean_x, mean_t, mean_s, log_var_x, log_var_t, log_var_s):
        mu_t, mu_s = torch.zeros(32), torch.ones(32)
        mu_t[10] = 1.
        mu_s[10] = 0.
        
        if self.training:
            z1 = mean_t + torch.exp(log_var_t/2) @ torch.normal(mu_t, torch.eye(32)).cuda()
            z2 = mean_s + torch.exp(log_var_s/2) @ torch.normal(mu_s, torch.eye(32)).cuda()
            std = log_var_x.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mean_x).cuda()
            
            return z, z1, z2
        else:
            return mean_x, mean_t, mean_s
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim = 1, input_shape = 32):
        super(Encoder, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(int(input_shape/4 * input_shape/4 * 16), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)

        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # W X H
        conv1 = self.relu(self.bn1(self.conv1(x)))
        # W/2 X H/2
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        # W/2 X H/2
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        # W/4 X H/4
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, int(self.input_shape/4 * self.input_shape/4 * 16))

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mu, logvar = self.fc21(fc1), self.fc22(fc1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

    
class Decoder(nn.Module):
    def __init__(self, input_dim = 1, input_shape = 32):
        super(Decoder, self).__init__()
        
        self.input_shape = input_shape
        
#         self.conv0 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.ConvTranspose2d(16, input_dim, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(16, 32, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)

        

        self.fc2 = nn.Linear(512, int(input_shape/4 * input_shape/4 * 16))
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(64, 512)

        self.relu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)

    def forward(self, x):
        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x).view(-1, 16, int(self.input_shape/4), int(self.input_shape/4))
        
        conv4 = self.relu(self.bn4(self.conv4(x)))
        conv3 = self.relu(self.bn3(self.conv3(conv4)))
        conv2 = self.relu(self.bn2(self.conv2(conv3)))
        x = self.relu(self.bn1(self.conv1(conv2)))
#         x = self.relu(self.bn1(self.conv0(conv1)))

        return x

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]
    
    
class Encoder_img(nn.Module):
    def __init__(self, nc = 3, ndf = 64, latent_variable_size = 1024):
        super(Encoder_img, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 7, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 7, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 7, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 7, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 7, 2, 0)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*3*3, latent_variable_size)
        self.fc_bn1 = nn.BatchNorm1d(latent_variable_size)
        self.fc2 = nn.BatchNorm1d(latent_variable_size)
  
        self.fc21 = nn.Linear(latent_variable_size, latent_variable_size)
        self.fc22 = nn.Linear(latent_variable_size, latent_variable_size)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*3*3)
    
        fc1 = self.relu(self.fc_bn1(self.fc1(h5)))
        fc2 = self.relu(self.fc2(fc1))
        
        mu, logvar = self.fc21(fc2), self.fc22(fc2)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]
    
    
class Decoder_img(nn.Module):
    def __init__(self, nc = 3, ngf = 64, latent_variable_size = 1024):
        super(Decoder_img, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.latent_variable_size = latent_variable_size

        # decoder
        self.dense1 = nn.Linear(latent_variable_size, ngf*8*2*3*3)

        self.d1 = nn.ConvTranspose2d(ngf*8*2,
                                       ngf*8,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1)
        self.bn1 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.d2 = nn.ConvTranspose2d(ngf*8,
                               ngf*4,
                               kernel_size=7,
                               stride = 2,
                               padding=1,
                               output_padding=1)
        self.bn2 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.d3 = nn.ConvTranspose2d(ngf*4,
                                       ngf*2,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1)
        self.bn3 = nn.BatchNorm2d(ngf*2, 1.e-3)
        
        self.d4 = nn.ConvTranspose2d(ngf*2,
                                       ngf,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=0,
                                       output_padding=0)
        self.bn4 = nn.BatchNorm2d(ngf, 1.e-3)
        
        self.d5 = nn.ConvTranspose2d(ngf,
                                       ngf,
                                       kernel_size=7,
                                       stride = 2,
                                       padding=0,
                                       output_padding=1)
        self.bn5 = nn.BatchNorm2d(ngf, 1.e-3)

        self.d6 = nn.Conv2d(ngf, out_channels= 3,
                                    kernel_size= 3, padding= 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.dense1(z).view(-1, self.ngf*8*2, 3, 3)
        h1 = self.leakyrelu(self.bn1(self.d1(z)))
        h2 = self.leakyrelu(self.bn2(self.d2(h1)))
        h3 = self.leakyrelu(self.bn3(self.d3(h2)))
        h4 = self.leakyrelu(self.bn4(self.d4(h3)))
        h5 = self.leakyrelu(self.bn5(self.d5(h4)))
        

        return self.sigmoid(self.d6(h5))
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]


class Mine(nn.Module):
# https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/GAN_VDB_MINE.ipynb    
    def __init__(self, noise_size=3, sample_size=2, output_size=1, hidden_size=128):
        super().__init__()
        self.fc1_noise = nn.Linear(noise_size, hidden_size, bias=False)
        self.fc1_sample = nn.Linear(sample_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.ma_et = None
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                
    def forward(self, noise, sample):
        x_noise = self.fc1_noise(noise)
        x_sample = self.fc1_sample(sample)
        x = F.leaky_relu(x_noise + x_sample + self.fc1_bias, negative_slope=2e-1)
        x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
        x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
        return x
    
class Encoder_Res(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder_Res, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, 2*hdim)           
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
class Decoder_Res(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Decoder_Res, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        cc = channels[-1]
        self.fc = nn.Sequential(
                      nn.Linear(hdim, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        
        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
                    
    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]    
    
    
class SD_VAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(SD_VAE, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]

        #Encoder Part
        self.encoder_i = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_i.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_i = nn.Linear((cc)*4*4, 2*hdim)

        self.encoder_r = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_r.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_r = nn.Linear((cc)*4*4, 2*hdim)
        
        #Projector Part
        self.projector = nn.Sequential(
            nn.Linear(2 * hdim, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, 2 * hdim)
        )
        
        #Decoder Part
        cc = channels[-1]
        self.fc2 = nn.Sequential(
                      nn.Linear(hdim * 2, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        self.decoder = nn.Sequential()
        for ch in channels[::-1]:
            self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.decoder.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.decoder.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))    
        
    def encode(self, x):        
        y_i = self.encoder_i(x).view(x.size(0), -1)
        y_i = self.fc1_i(y)
        mu_i, logvar_i = y_i.chunk(2, dim=1)
        z_i = self.reparameterize(mu_i, logvar_i)
        
        y_r = self.encoder_r(x).view(x.size(0), -1)
        y_r = self.fc1_r(y)
        mu_r, logvar_r = y_r.chunk(2, dim=1)
        z_r = self.reparameterize(mu_r, logvar_r)
        
        return z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def project(self, z):
        z = projector(z)
        return z
    
    def decode(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc2(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.main(y)
        return y
    
    def forward(self, x):
        z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = self.encoder(x)
        
        z = torch.cat([z_i, z_r], dim = 1)
        z = self.project(z)
        recon = self.decode(z)
        
        return recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)

class Projector_sep(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Projector_sep, self).__init__()
        
        self.projector_x = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.projector_y = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.projector_r = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.projector_s = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
        
    def forward(self, x):
        return self.projector_x(x), self.projector_y(x), self.projector_r(x), self.projector_s(x)
    
class Projector(nn.Module):
    def __init__(self, dim):
        super(Projector, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return self.projector(x)
    
class SD_VAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(SD_VAE, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]

        #Encoder Part
        self.encoder_i = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_i.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_i.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_i = nn.Linear((cc)*4*4, 2*hdim)

        self.encoder_r = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.encoder_r.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.encoder_r.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc1_r = nn.Linear((cc)*4*4, 2*hdim)
        
        #Projector Part
        self.projector = nn.Sequential(
            nn.Linear(2 * hdim, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, 2 * hdim),
        )
        
        #Decoder Part
        cc = channels[-1]
        self.fc2 = nn.Sequential(
                      nn.Linear(hdim * 2, cc*4*4),
                      nn.ReLU(True),
                  )
                  
        sz = 4
        self.decoder = nn.Sequential()
        for ch in channels[::-1]:
            self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.decoder.add_module('up_to_{}'.format(sz*2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz*2
       
        self.decoder.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.decoder.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))    
        
    def encode(self, x):        
        y_i = self.encoder_i(x).view(x.size(0), -1)
        y_i = self.fc1_i(y_i)
        mu_i, logvar_i = y_i.chunk(2, dim=1)
        z_i = self.reparameterize(mu_i, logvar_i)
        
        y_r = self.encoder_r(x).view(x.size(0), -1)
        y_r = self.fc1_r(y_r)
        mu_r, logvar_r = y_r.chunk(2, dim=1)
        z_r = self.reparameterize(mu_r, logvar_r)
        
        return z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def project(self, z):
        z = self.projector(z)
        return z
    
    def decode(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc2(z)
        y = y.view(z.size(0), -1, 4, 4)
        y = self.decoder(y)
        return y
    
    def forward(self, x):
        z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = self.encode(x)
        z = torch.cat([z_i, z_r], dim = 1)
        z = self.project(z)
        recon = self.decode(z)
        
        return recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)

class Project(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Project, self).__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
        
    def forward(self, x):
        return self.projector(x)
    
class Encoder_FADES(nn.Module):
    def __init__(self, cdim=3, hdim=512, feat_dim = 32, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder_FADES, self).__init__() 
        
        assert (2 ** len(channels)) * 4 == image_size
        
        self.hdim = hdim
        cc = channels[0]
        self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),                
                nn.AvgPool2d(2),
              )
              
        sz = image_size//2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz//2), nn.AvgPool2d(2))
            cc, sz = ch, sz//2
        
        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))                    
        self.fc = nn.Linear((cc)*4*4, hdim)
        
        self.proj_x = Project(hdim, 2*(hdim - 3 * feat_dim))
        self.proj_y = Project(hdim, feat_dim * 2)
        self.proj_r = Project(hdim, feat_dim * 2)
        self.proj_s = Project(hdim, feat_dim * 2)
    
    def forward(self, x):        
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        
#         mu, logvar = y.chunk(2, dim=1)
        mu_x, logvar_x = self.proj_x(y).chunk(2, dim=1)
        mu_y, logvar_y = self.proj_y(y).chunk(2, dim=1)
        mu_r, logvar_r = self.proj_r(y).chunk(2, dim=1)
        mu_s, logvar_s = self.proj_s(y).chunk(2, dim=1)
        
        z_x = self.reparameterize(mu_x, logvar_x)
        z_y = self.reparameterize(mu_y, logvar_y)
        z_r = self.reparameterize(mu_r, logvar_r)
        z_s = self.reparameterize(mu_s, logvar_s)
        
        return (z_x, z_y, z_r, z_s), (mu_x, mu_y, mu_r, mu_s), (logvar_x, logvar_y, logvar_r, logvar_s)
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]   
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() 
        
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
class _Residual_Block(nn.Module): 
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()
        
        midc=int(outc*scale)
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
#         output = self.relu2(self.bn2(torch.add(output,identity_data)))
        return output 
