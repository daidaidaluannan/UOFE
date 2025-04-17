import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from model import VAE, VAE_vessel  # 假设 VAE 和 VAE_vessel 都在 model.py 文件中
import argparse
from tqdm import tqdm

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="Extract latent features from images using a pre-trained VAE model.")
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained VAE model.')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing images.')
parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file.')
parser.add_argument('--latent_dim', type=int, default=128, required=True, help='Dimensionality of the latent space.')
parser.add_argument('--model_type', type=str, default='VAE', choices=['VAE', 'VAE_vessel'], help='Type of VAE model to use (VAE or VAE_vessel).')

args = parser.parse_args()

model_path = args.model_path
image_folder = args.image_folder
output_csv = args.output_csv
latent_dim = args.latent_dim
model_type = args.model_type

# 定义超参数和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载预训练的模型，根据 model_type 参数选择模型
if model_type == 'VAE':
    vae = VAE(latent_dim=latent_dim).to(device)
elif model_type == 'VAE_vessel':
    vae = VAE_vessel(latent_dim=latent_dim).to(device)

# 加载模型权重
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.eval()

# 初始化结果列表
results = []

# 遍历图像文件夹中的所有图像
for filename in tqdm(os.listdir(image_folder)):
    if filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        image_id = filename.split('_')[0]  # 提取图像ID
        
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # 提取潜在变量
        with torch.no_grad():
            _, mu, logvar = vae(image)
            z = vae.reparameterize(mu, logvar)
        
        # 将图像ID和潜在变量添加到结果列表
        result = [image_id] + z.squeeze().tolist()
        results.append(result)

# 将结果保存到CSV文件
df = pd.DataFrame(results, columns=['eid'] + [f'latent_{i+1}' for i in range(latent_dim)])
df.to_csv(output_csv, index=False)

print(f"Latent features saved to {output_csv}")
