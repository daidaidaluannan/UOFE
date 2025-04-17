import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
from model import VGG_16,VAE_new,VAE_vessel, VAE_skip,VAEAttn,VAE_vessel_Res,VAE_diff,AutoencoderKL_diff
import argparse

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="Train VAE on fundus images.")
parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--model_path', type=str, default=None, help='Path to a pretrained model (optional)')
args = parser.parse_args()

latent_dim = args.latent_dim
num_epochs = args.num_epochs
model_path = args.model_path


#latent_dim = 64
#num_epochs = 200
save_dir = "/haizhu_data_8T/data_wcy/code/vae_modle/result/vessel_kaggle_vae_dim128_AutoencoderKL_diff/"
train_dir = '/haizhu_data_8T/data_wcy/code/vae_modle/train_data/binary_vessel/train/'
val_dir = '/haizhu_data_8T/data_wcy/code/vae_modle/train_data/binary_vessel/val/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 定义数据集类
class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集目录，包含所有图像
            transform (callable, optional): 图像预处理变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpeg') or filename.endswith('.png'):
                self.image_paths.append(os.path.join(root_dir, filename))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
])

transform_vessel = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# 加载数据集
train_dataset = FundusDataset(root_dir= train_dir, transform=transform_vessel)
val_dataset = FundusDataset(root_dir= val_dir, transform=transform_vessel)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

####################################################
# 定义编码器
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
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
####################################################
    
# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.sum(kl_loss)
    return recon_loss + kl_loss


# 初始化模型、优化器

#model = AutoencoderKL_diff(latent_dim=latent_dim).to(device)
model = AutoencoderKL_diff().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)


# 加载预训练模型（如果指定了路径）
start_epoch = 1
if model_path is not None:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
    min_val_loss = checkpoint['val_loss']
    print(f'Loaded model from {model_path}, resuming training from epoch {start_epoch}')
else:
    min_val_loss = float('inf')


# 训练模型

train_losses = []
val_losses = []

import torch
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np


# 日志文件路径
log_file = os.path.join(save_dir, "training_log.txt")

# 初始化日志文件
with open(log_file, "w") as f:
    f.write("Epoch\tTrain_Loss\tValidation_Loss\tBest_Model_Saved\n")


# 训练和验证循环
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item()/len(data):.4f}')
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    print(f'====> Epoch: {epoch} Average validation loss: {avg_val_loss:.4f}')

    # 检查当前验证损失是否是最小的
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
        print(f'====> New best model saved with validation loss: {avg_val_loss:.4f}')
        best_model_saved = True
    
    # 保存日志信息到文件
    with open(log_file, "a") as f:
        f.write(f"{epoch}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{best_model_saved}\n")
    
   ## 每10个epoch保存一次模型
   #if epoch % 10 == 0:
   #    torch.save(model.state_dict(), f'{save_dir}/epoch_{epoch}.pth')
        
    # 可视化重建结果
    with torch.no_grad():
        sample = next(iter(val_loader))
        sample = sample.to(device)
        recon_sample, _, _ = model(sample)
        comparison = torch.cat([sample[:8], recon_sample[:8]])
        comparison = comparison.cpu()
        grid = utils.make_grid(comparison, nrow=8, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(15, 5))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.title(f'Epoch {epoch} Reconstruction')
        plt.axis('off')
        plt.savefig(f'{save_dir}/Epoch_{epoch}.png')


# 绘制训练和验证损失曲线
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(save_dir + 'training_validation_loss.png')
