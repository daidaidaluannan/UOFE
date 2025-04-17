# model.py

import torch
import torch.nn as nn
import torchvision.models as models
from diffusers import AutoencoderKL

class Encoder_vgg16(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_vgg16, self).__init__()
        self.latent_dim = latent_dim
        self.vgg16 = models.vgg16(pretrained=True).features
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)
    
    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder_vgg16(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_vgg16, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class VGG_16(nn.Module):
    def __init__(self, latent_dim=128, device=None):
        super(VGG_16, self).__init__()
        self.encoder = Encoder_vgg16(latent_dim)
        self.decoder = Decoder_vgg16(latent_dim)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

###################################################################################################
class Encoder_vae(nn.Module):
    def __init__(self, in_ch, nf, latent_dim):
        super(Encoder_vae, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, nf, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),  # 128x128
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1),  # 64x64
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),  # 32x32
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),  # 8x8
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),  # 4x4
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 2x2
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),  # 2x2
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Conv2d(nf * 8, nf * 8, kernel_size=2, stride=1, padding=0),  # 1x1
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(nf * 8, latent_dim)
        self.fc_logvar = nn.Linear(nf * 8, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

#class Decoder_vae(nn.Module):
#    def __init__(self, out_ch, nf, latent_dim, is_binary):
#        super(Decoder_vae, self).__init__()
#        self.decoder_input = nn.Linear(latent_dim, nf * 8)
#        
#        self.decoder = nn.Sequential(
#            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 2x2
#            nn.BatchNorm2d(nf * 8),
#            nn.LeakyReLU(0.2),
#            
#            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 4x4
#            nn.BatchNorm2d(nf * 8),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.5),
#            
#            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 8x8
#            nn.BatchNorm2d(nf * 8),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.5),
#            
#            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 16x16
#            nn.BatchNorm2d(nf * 8),
#            nn.LeakyReLU(0.2),
#            nn.Dropout(0.5),
#            
#            nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=3, stride=2, padding=1),  # 32x32
#            nn.BatchNorm2d(nf * 4),
#            nn.LeakyReLU(0.2),
#            nn.ConvTranspose2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),  # 32x32
#            nn.BatchNorm2d(nf * 4),
#            nn.LeakyReLU(0.2),
#            
#            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=3, stride=2, padding=1),  # 64x64
#            nn.BatchNorm2d(nf * 2),
#            nn.LeakyReLU(0.2),
#            nn.ConvTranspose2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1),  # 64x64
#            nn.BatchNorm2d(nf * 2),
#            nn.LeakyReLU(0.2),
#            
#            nn.ConvTranspose2d(nf * 2, nf, kernel_size=3, stride=2, padding=1),  # 128x128
#            nn.BatchNorm2d(nf),
#            nn.LeakyReLU(0.2),
#            nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=1, padding=1),  # 128x128
#            nn.BatchNorm2d(nf),
#            nn.LeakyReLU(0.2),
#            
#            nn.ConvTranspose2d(nf, out_ch, kernel_size=3, stride=2, padding=1),  # 256x256
#            nn.Tanh() if not is_binary else nn.Sigmoid()
#        )
#    
#    def forward(self, z):
#        x = self.decoder_input(z)
#        x = x.view(-1, 256, 1, 1)
#        x = self.decoder(x)
#        return x

class Decoder_vae(nn.Module):
    def __init__(self, out_ch, nf, latent_dim, is_binary):
        super(Decoder_vae, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, nf * 8 * 2 * 2)  # 注意：这里调整为适合解码器的输入形状

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2x2 -> 4x4
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(nf * 2, nf, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 -> 128x128
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(nf, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128 -> 256x256
        )

        # 最后一层的激活函数
        self.final_activation = nn.Tanh() if not is_binary else nn.Sigmoid()

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 2, 2)  # 注意：这里调整为 (batch_size, nf * 8, 2, 2)
        x = self.decoder(x)
        x = self.final_activation(x)
        return x


class VAE_new(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=32, latent_dim=30, is_binary=False, device=None):
        super(VAE_new, self).__init__()
        self.encoder = Encoder_vae(in_ch, nf, latent_dim)
        self.decoder = Decoder_vae(out_ch, nf, latent_dim, is_binary)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
###################################################################################################

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(1024 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 7 * 7, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, 1024 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 7, 7)
        x = self.decoder(x)
        return x

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, latent_dim=128,device=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
###################################################################################################
class Encoder_vessel(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_vessel, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(1024 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 7 * 7, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# 定义解码器
class Decoder_vessel(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_vessel, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, 1024 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 7, 7)
        x = self.decoder(x)
        return x

# 定义VAE模型
class VAE_vessel(nn.Module):
    def __init__(self, latent_dim=128,device=None):
        super(VAE_vessel, self).__init__()
        self.encoder = Encoder_vessel(latent_dim)
        self.decoder = Decoder_vessel(latent_dim)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
###################################################################################################
import torch
import torch.nn as nn

class Encoder_skip(nn.Module):
    def __init__(self, in_ch, nf, latent_dim):
        super(Encoder_skip, self).__init__()
        self.latent_dim = latent_dim
        
        # 编码器块
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, nf, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.enc6 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.enc7 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),  # 2x2
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.enc8 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=2, stride=1, padding=0),  # 1x1
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.fc_mu = nn.Linear(nf * 8, latent_dim)
        self.fc_logvar = nn.Linear(nf * 8, latent_dim)
    
    def forward(self, x):
        # 保存中间特征用于跳跃连接
        features = []
        
        x1 = self.enc1(x)
        features.append(x1)
        
        x2 = self.enc2(x1)
        features.append(x2)
        
        x3 = self.enc3(x2)
        features.append(x3)
        
        x4 = self.enc4(x3)
        features.append(x4)
        
        x5 = self.enc5(x4)
        features.append(x5)
        
        x6 = self.enc6(x5)
        features.append(x6)
        
        x7 = self.enc7(x6)
        features.append(x7)
        
        x8 = self.enc8(x7)
        features.append(x8)
        
        x = x8.view(x8.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar, features

class Decoder_skip(nn.Module):
    def __init__(self, out_ch, nf, latent_dim, is_binary):
        super(Decoder_skip, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, nf * 8)
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2x2
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8 * 2, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8 * 2, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8 * 2, nf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2)
        )
        
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(nf * 8 * 2, nf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2)
        )
        
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(nf * 4 * 2, nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2)
        )
        
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2 * 2, nf, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2)
        )
        
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x256
            nn.Tanh() if not is_binary else nn.Sigmoid()
        )
    
    def forward(self, z, encoder_features):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 1, 1)
        
        x = self.dec1(x)
        x = torch.cat([x, encoder_features[-2]], dim=1)  # 跳跃连接
        
        x = self.dec2(x)
        x = torch.cat([x, encoder_features[-3]], dim=1)
        
        x = self.dec3(x)
        x = torch.cat([x, encoder_features[-4]], dim=1)
        
        x = self.dec4(x)
        x = torch.cat([x, encoder_features[-5]], dim=1)
        
        x = self.dec5(x)
        x = torch.cat([x, encoder_features[-6]], dim=1)
        
        x = self.dec6(x)
        x = torch.cat([x, encoder_features[-7]], dim=1)
        
        x = self.dec7(x)
        x = torch.cat([x, encoder_features[-8]], dim=1)
        
        x = self.dec8(x)
        
        return x
    
    

class VAE_skip(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=32, latent_dim=64, is_binary=False, device=None):
        super(VAE_skip, self).__init__()
        self.encoder = Encoder_skip(in_ch, nf, latent_dim)
        self.decoder = Decoder_skip(out_ch, nf, latent_dim, is_binary)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar, encoder_features = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, encoder_features)
        return recon_x, mu, logvar
###################################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

############################################
# 1. 自注意力模块 (2D)
############################################
class SelfAttn2d(nn.Module):
    """
    简易的 2D 自注意力模块，输入形状 (B, C, H, W).
    参考 Self-Attention GAN 的实现思路。
    """
    def __init__(self, in_dim):
        super(SelfAttn2d, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,       kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) Q, K, V
        query = self.query_conv(x)  # (B, C//8, H, W)
        key   = self.key_conv(x)    # (B, C//8, H, W)
        value = self.value_conv(x)  # (B, C,    H, W)

        # 2) reshape
        #    Q -> (B, Cq, N), K -> (B, Ck, N), V -> (B, Cv, N)
        #    其中 N = H*W
        query = query.view(B, -1, H*W)    # (B, C//8, N)
        key   = key.view(B, -1, H*W)      # (B, C//8, N)
        value = value.view(B, -1, H*W)    # (B, C,    N)

        # 3) 计算注意力 attn = softmax(Q^T * K)
        #    Q^T -> (B, N, Cq), K -> (B, Ck, N)
        #    => attn shape (B, N, N)
        attn = torch.bmm(query.permute(0, 2, 1), key)
        attn = F.softmax(attn, dim=-1)

        # 4) 加权 V
        #    (B, N, N) x (B, N, Cv) => (B, N, Cv)
        out = torch.bmm(attn, value.permute(0, 2, 1))
        out = out.permute(0, 2, 1).view(B, C, H, W)

        # 5) 残差
        out = self.gamma * out + x
        return out

############################################
# 2. 编码器 (无 skip-connection, 下采样到 4096)
############################################
class EncoderAttn(nn.Module):
    """
    假设输入图像 256×256，nf=32，最后得到 (B, 256, 4,4) => flatten=4096
    再做  Linear(4096, latent_dim) 用于 mu, logvar
    """
    def __init__(self, in_ch=3, nf=32, latent_dim=64):
        super(EncoderAttn, self).__init__()
        self.latent_dim = latent_dim

        # 下采样共 6 层，最终到 H=4, W=4, C=256 => flatten=4096
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, nf, 4, 2, 1),  # 256->128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, 4, 2, 1),   # 128->64
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, 4, 2, 1), # 64->32
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(nf*4, nf*8, 4, 2, 1), # 32->16
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(nf*8, nf*8, 4, 2, 1), # 16->8
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttn2d(nf*8)  # 在 8×8 这一层做一次注意力
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(nf*8, nf*8, 4, 2, 1), # 8->4
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttn2d(nf*8)  # 在 4×4 这一层做一次注意力
        )

        # 线性层
        self.fc_mu = nn.Linear(nf*8*4*4, latent_dim)      # 4096->latent_dim
        self.fc_lv = nn.Linear(nf*8*4*4, latent_dim)      # 4096->latent_dim

    def forward(self, x):
        """
        x: (B, 3, 256, 256)
        返回 mu, logvar
        """
        x = self.enc1(x)  # (B, 32,128,128)
        x = self.enc2(x)  # (B, 64, 64,64)
        x = self.enc3(x)  # (B,128, 32,32)
        x = self.enc4(x)  # (B,256, 16,16)
        x = self.enc5(x)  # (B,256, 8,8)
        x = self.enc6(x)  # (B,256, 4,4)
        
        # flatten
        #x = x.view(x.size(0), -1)  # (B, 256*4*4=4096)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_lv(x)
        return mu, logvar

############################################
# 3. 解码器 (无 skip-connection, 带自注意力)
############################################
class DecoderAttn(nn.Module):
    """
    只用潜在向量 z，就能上采样回 256×256，并在若干层插入注意力。
    """
    def __init__(self, latent_dim=64, nf=32, out_ch=3, is_binary=False):
        super(DecoderAttn, self).__init__()

        self.init_fc = nn.Linear(latent_dim, nf*8*4*4)  # => (B, 256*4*4)=4096

        # 依次上采样回 256×256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(nf*8, nf*8, 4, 2, 1),  # 4->8
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttn2d(nf*8)  # 在 8×8 做一次注意力
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(nf*8, nf*8, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1),  # 16->32
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttn2d(nf*4)  # 在 32×32 做一次注意力
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1),  # 32->64
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, nf, 4, 2, 1),    # 64->128
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(nf, out_ch, 4, 2, 1),  # 128->256
            nn.Tanh() if not is_binary else nn.Sigmoid()
        )

    def forward(self, z):
        """
        z: shape (B, latent_dim)
        """
        B = z.size(0)
        # 1) 先投影到 4×4, channel=256
        x = self.init_fc(z)               # => (B, 256*4*4)
        x = x.view(B, -1, 4, 4)           # => (B, 256, 4,4)

        # 2) 逐层上采样 + 自注意力
        x = self.up1(x)  # => 8×8
        x = self.up2(x)  # => 16×16
        x = self.up3(x)  # => 32×32
        x = self.up4(x)  # => 64×64
        x = self.up5(x)  # => 128×128
        x = self.up6(x)  # => 256×256

        return x

############################################
# 4. VAE: 组合 EncoderAttn & DecoderAttn
############################################
class VAEAttn(nn.Module):
    def __init__(self, 
                 in_ch=3, out_ch=3, 
                 nf=32, latent_dim=64, 
                 is_binary=False, device=None):
        super(VAEAttn, self).__init__()
        self.encoder = EncoderAttn(in_ch, nf, latent_dim)
        self.decoder = DecoderAttn(latent_dim, nf, out_ch, is_binary)
        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
##########################################################################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class Encoder_vessel_Res(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_vessel_Res, self).__init__()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56x56
        )
        
        # 残差块
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )  # 56x56
        
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128)
        )  # 28x28
        
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )  # 14x14
        
        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            ResBlock(512, 512)
        )  # 7x7
        
        self.layer5 = nn.Sequential(
            ResBlock(512, 1024, stride=2),
            ResBlock(1024, 1024)
        )  # 4x4
        
        # 全连接层
        self.fc_mu = nn.Linear(1024 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 4 * 4, latent_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.initial(x)
        
        x = self.layer1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder_vessel_Res(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_vessel_Res, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 1024 * 4 * 4)
        
        self.layer1 = nn.Sequential(
            ResBlock(1024, 1024),
            ResBlock(1024, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # 8x8
        )
        
        self.layer2 = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)  # 16x16
        )
        
        self.layer3 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)  # 32x32
        )
        
        self.layer4 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 64x64
        )
        
        self.layer5 = nn.Sequential(
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 128x128
        )
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        
        x = self.layer1(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        x = self.final(x)
        
        return x

class VAE_vessel_Res(nn.Module):
    def __init__(self, latent_dim=256, device=None):
        super(VAE_vessel_Res, self).__init__()
        self.encoder = Encoder_vessel_Res(latent_dim)
        self.decoder = Decoder_vessel_Res(latent_dim)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
##########################################################################################
# 定义编码器（使用 Stable Diffusion 预训练的 VAE 编码器）
class Encoder_diff(nn.Module):
    def __init__(self, latent_dim, device="cuda"):
        super(Encoder_diff, self).__init__()
        self.device = device
        self.latent_dim = latent_dim

        # 加载 Stable Diffusion 预训练 VAE
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae = AutoencoderKL.from_pretrained("/haizhu_data_8T/data_wcy/code/vae_modle/diffusion_pytorch_model/").to(device)

    #def forward(self, x):
    #    """ 使用 Stable Diffusion VAE 进行编码，返回 mu 和 logvar """
    #    with torch.no_grad():
    #        latent = self.vae.encode(x).latent_dist  # 获取潜在分布
    #        mu, logvar = latent.mean, latent.logvar  # 提取均值和方差
    #    return mu, logvar

    def forward(self, x):
        
        latent = self.vae.encode(x).latent_dist  # 获取潜在分布
        mu, logvar = latent.mean, latent.logvar  # 提取均值和方差
        return mu, logvar

# 定义解码器（使用 Stable Diffusion 预训练的 VAE 解码器）
class Decoder_diff(nn.Module):
    def __init__(self, latent_dim, device="cuda"):
        super(Decoder_diff, self).__init__()
        self.device = device

        # 仍然使用 Stable Diffusion VAE 进行解码
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae = AutoencoderKL.from_pretrained("/haizhu_data_8T/data_wcy/code/vae_modle/diffusion_pytorch_model/").to(device)
        

    #def forward(self, z):
    #    """ 使用 Stable Diffusion VAE 进行解码 """
    #    with torch.no_grad():
    #        recon_x = self.vae.decode(z).sample  # 生成图像
    #    return recon_x
    
    def forward(self, z):
        
        recon_x = self.vae.decode(z).sample  # 生成图像
        return recon_x

# 定义 VAE 模型（与原始结构保持一致）
class VAE_diff(nn.Module):
    def __init__(self, latent_dim=4, device="cuda"):
        super(VAE_diff, self).__init__()
        self.device = device
        self.encoder = Encoder_diff(latent_dim, device)
        self.decoder = Decoder_diff(latent_dim, device)

    def reparameterize(self, mu, logvar):
        """ 变分重参数化 """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)  # 采样噪声
        return mu + eps * std  # 变分采样

    def forward(self, x):
        """ VAE 处理流程 """
        mu, logvar = self.encoder(x)  # 编码
        z = self.reparameterize(mu, logvar)  # 采样
        recon_x = self.decoder(z)  # 解码
        return recon_x, mu, logvar  # 返回重建图像、均值、方差
    

#####################################################################################################################
# ResNet Block
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nn.SiLU()  # SiLU 激活函数

        # Shortcut connection
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)

        return x + shortcut

# Downsampling Block
class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels),
            ResnetBlock2D(out_channels, out_channels)
        ])
        self.downsampler = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)  # 2× 下采样

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        x = self.downsampler(x)
        return x

# Upsampling Block
class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels),
            ResnetBlock2D(out_channels, out_channels)
        ])
        self.upsampler = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, output_padding=1)  # 2× 上采样

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        x = self.upsampler(x)
        return x

# Attention Block (用于 UNet 中的中间层)
class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # Flatten spatial dimensions
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (c ** 0.5), dim=-1)
        out = attn @ v
        out = self.to_out(out)
        return out.view(b, c, h, w)

# Encoder
class Encoder_KL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([
            DownEncoderBlock2D(128, 128),
            DownEncoderBlock2D(128, 256),
            DownEncoderBlock2D(256, 512),
            DownEncoderBlock2D(512, 512)
        ])
        self.mid_block = Attention(512)  # UNet 中间层
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, stride=1, padding=1)
        
        # 新增：将特征图转换为1维向量
        self.flatten = nn.Flatten()
        
        # 输出均值和对数方差，各128维
        self.fc_mu = nn.Linear(8 * (H//16) * (W//16), 128)  
        self.fc_logvar = nn.Linear(8 * (H//16) * (W//16), 128)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_out(x)
        
        # 将特征图转换为1维向量
        x = self.flatten(x)
        
        # 计算均值和对数方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # 重参数化技巧：z = mu + std * eps
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, logvar

# Decoder
class Decoder_KL(nn.Module):
    def __init__(self):
        super().__init__()
        # 新增：将1维向量转回特征图
        self.fc = nn.Linear(128, 4 * (H//16) * (W//16))  # H和W是输入图像的高宽
        self.unflatten = nn.Unflatten(1, (4, H//16, W//16))
        
        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1)
        self.up_blocks = nn.ModuleList([
            UpDecoderBlock2D(512, 512),
            UpDecoderBlock2D(512, 256),
            UpDecoderBlock2D(256, 128),
            UpDecoderBlock2D(128, 128)
        ])
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # 将1维向量转回特征图
        x = self.fc(z)
        x = self.unflatten(x)
        
        x = self.conv_in(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x

# AutoencoderKL
class AutoencoderKL_diff(nn.Module):
    def __init__(self, input_height=256, input_width=256):
        super().__init__()
        global H, W
        H, W = input_height, input_width
        
        self.encoder = Encoder_KL()
        self.decoder = Decoder_KL()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)  # 现在返回三个值
        recon_x = self.decoder(z)
        return recon_x, mu, logvar  # 返回三个值以匹配调用代码的期望
    
################################################################################
# 编码器保持类似，但不再输出均值和方差
class Encoder_vessel_VQ(nn.Module):
    def __init__(self):
        super(Encoder_vessel_VQ, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 64, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)  # 返回形状为 [B, 64, 7, 7] 的特征图

# 新增：向量量化层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        # 创建码本
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # 输入形状转换: [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # 将输入展平为 [B*H*W, C]
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # 计算与每个码本向量的平方欧氏距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # 找到最近的码本向量
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 获取量化后的向量
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # 直通估计器（Straight-Through Estimator）
        quantized = inputs + (quantized - inputs).detach()
        
        # 返回形状转换: [B, H, W, C] -> [B, C, H, W]
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices

# 解码器基本不变
class Decoder_vessel_VQ(nn.Module):
    def __init__(self):
        super(Decoder_vessel_VQ, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)

# 定义VQ-VAE模型
class VQVAE_vessel(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, device=None):
        super(VQVAE_vessel, self).__init__()
        self.device = device
        
        self.encoder = Encoder_vessel_VQ()
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder_vessel_VQ()
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vector_quantizer(z)
        x_recon = self.decoder(z_q)
        
        return x_recon, z, z_q, vq_loss, indices
################################################################################
# 编码器：将图像压缩为一维潜在向量
class Encoder_vessel_VQ_1D(nn.Module):
    def __init__(self):
        super(Encoder_vessel_VQ_1D, self).__init__()
        # 卷积编码层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU()
        )
        
        # 扁平化和全连接层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 7 * 7, 128)  # 压缩为128维向量
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        z = self.fc(x)
        # 保留原始一维向量用于损失计算
        z_2d = z.unsqueeze(2).unsqueeze(3)  # 形状 [B, 128, 1, 1]
        return z_2d, z

# 向量量化器：将连续向量映射到离散码本
class VectorQuantizer1D(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=128, commitment_cost=0.25):
        super(VectorQuantizer1D, self).__init__()
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        
        # 创建码本
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # inputs已经是形状 [B, 128] 的一维向量
        inputs_flat = inputs
        
        # 计算与每个码本向量的平方欧氏距离
        distances = (torch.sum(inputs_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs_flat, self._embedding.weight.t()))
            
        # 找到最近的码本向量
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 创建one-hot编码
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # 获取量化后的向量
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_flat)
        q_latent_loss = F.mse_loss(quantized, inputs_flat.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized_st = inputs_flat + (quantized - inputs_flat).detach()
        
        # 返回量化后的向量和损失
        quantized_2d = quantized_st.unsqueeze(2).unsqueeze(3)  # 形状 [B, 128, 1, 1]
        
        return quantized_2d, quantized_st, loss, encoding_indices

# 解码器：将一维潜在向量重建为图像
class Decoder_vessel_VQ_1D(nn.Module):
    def __init__(self):
        super(Decoder_vessel_VQ_1D, self).__init__()
        
        # 全连接层将一维向量映射回特征图
        self.fc = nn.Linear(128, 512 * 7 * 7)
        
        # 转置卷积层
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Tanh()
        )
    
    def forward(self, x, x_flat=None):
        # 如果输入是2D特征图 [B, 128, 1, 1]
        if x_flat is None:
            x_flat = x.squeeze(-1).squeeze(-1)  # [B, 128]
        
        x = self.fc(x_flat)  # [B, 512*7*7]
        x = x.view(-1, 512, 7, 7)  # [B, 512, 7, 7]
        x = self.deconv_layers(x)  # [B, 3, 224, 224]
        return x

# 完整的VQ-VAE模型
class VQVAE_vessel_1D(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, device=None):
        super(VQVAE_vessel_1D, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        self.encoder = Encoder_vessel_VQ_1D()
        self.vector_quantizer = VectorQuantizer1D(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder_vessel_VQ_1D()
    
    def forward(self, x):
        # 编码
        z_2d, z = self.encoder(x)
        
        # 向量量化
        z_q_2d, z_q, vq_loss, indices = self.vector_quantizer(z)
        
        # 解码
        x_recon = self.decoder(z_q_2d, z_q)
        
        return x_recon, z, z_q, vq_loss, indices
################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + self.pos_embed
        return x  # (B, N, embed_dim)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ViT AutoEncoder
class ViTAutoEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=128, depth=8, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Decoder uses same number of tokens
        self.decoder_tokens = (img_size // patch_size) ** 2
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.decoder_tokens, embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)

        # Encoder
        for blk in self.encoder_blocks:
            x = blk(x)

        # Average pooling to get endophenotype
        endophenotype = x.mean(dim=1)  # (B, D)

        # Expand back to token sequence for decoder
        x = endophenotype.unsqueeze(1).repeat(1, self.decoder_tokens, 1)  # (B, N, D)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        # Map tokens back to patches
        x = self.output_proj(x)  # (B, N, patch_size^2)
        x = x.permute(0, 2, 1)  # (B, patch_size^2, N)
        x = F.fold(
            x,
            output_size=(self.img_size, self.img_size),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        return x, endophenotype
    
###############################################################################
# Patch Embedding Layer with [CLS] token
#class PatchEmbedding(nn.Module):
#    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=128):
#        super().__init__()
#        self.img_size = img_size
#        self.patch_size = patch_size
#        self.n_patches = (img_size // patch_size) ** 2
#
#        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
#
#        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # ✅ [CLS] token
#        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))  # ✅ +1 for CLS
#
#    def forward(self, x):
#        B = x.size(0)
#        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
#        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
#
#        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
#        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)
#        x = x + self.pos_embed
#        return x  # (B, N+1, embed_dim)
#
## Transformer Block
#class TransformerBlock(nn.Module):
#    def __init__(self, embed_dim=128, num_heads=8, mlp_ratio=2.0):
#        super().__init__()
#        self.norm1 = nn.LayerNorm(embed_dim)
#        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#        self.norm2 = nn.LayerNorm(embed_dim)
#        self.mlp = nn.Sequential(
#            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
#            nn.GELU(),
#            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
#        )
#
#    def forward(self, x):
#        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#        x = x + self.mlp(self.norm2(x))
#        return x
#
## ViT AutoEncoder with CLS token and Projection Head
#class ViTAutoEncoder(nn.Module):
#    def __init__(self, img_size=224, patch_size=8, in_channels=1, embed_dim=128, depth=8, num_heads=8):
#        super().__init__()
#        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#        self.encoder_blocks = nn.ModuleList([
#            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
#        ])
#
#        # ✅ Projection head to refine CLS token --> endophenotype
#        self.projection_head = nn.Sequential(
#            nn.LayerNorm(embed_dim),
#            nn.Linear(embed_dim, embed_dim),
#            nn.GELU(),
#            nn.Linear(embed_dim, embed_dim)
#        )
#
#        # Decoder
#        self.decoder_tokens = (img_size // patch_size) ** 2
#        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.decoder_tokens, embed_dim))
#        self.decoder_blocks = nn.ModuleList([
#            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
#        ])
#        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size)
#
#        self.img_size = img_size
#        self.patch_size = patch_size
#
#    def forward(self, x):
#        B = x.size(0)
#        x = self.patch_embed(x)  # (B, N+1, D)
#
#        # Encoder
#        for blk in self.encoder_blocks:
#            x = blk(x)
#
#        # ✅ Use CLS token as endophenotype
#        cls_token = x[:, 0]  # (B, D)
#        endophenotype = self.projection_head(cls_token)  # (B, D)
#
#        # Expand CLS to decoder token sequence
#        x = endophenotype.unsqueeze(1).repeat(1, self.decoder_tokens, 1)  # (B, N, D)
#        x = x + self.decoder_pos_embed
#
#        for blk in self.decoder_blocks:
#            x = blk(x)
#
#        # Reconstruct patches
#        x = self.output_proj(x)  # (B, N, patch_size^2)
#        x = x.permute(0, 2, 1)  # (B, patch_size^2, N)
#        x = F.fold(
#            x,
#            output_size=(self.img_size, self.img_size),
#            kernel_size=self.patch_size,
#            stride=self.patch_size
#        )
#
#        return x, endophenotype