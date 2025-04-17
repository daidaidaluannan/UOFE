import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import ViTAutoEncoder  # 请确保你能导入你的模型定义
from tqdm import tqdm

# ========================
# 参数设置
# ========================
image_dir = "/home/wcy/data/UKB/ukb_eye/Results_right/M2/binary_vessel/raw/"  # 替换为你的图像文件夹路径
model_path = "/home/wcy/data/UKB/eye_feature/model/vessel_ViTAutoEncoder.pth"  # 替换为你的模型权重路径
output_csv = "/home/wcy/data/UKB/eye_feature/feature_data/right_vit.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 图像预处理
# ========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========================
# 加载模型
# ========================
model = ViTAutoEncoder(img_size=224, patch_size=16, embed_dim=128, depth=8, num_heads=8).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========================
# 遍历图像并提取潜在变量
# ========================
results = []

image_files = sorted([f for f in os.listdir(image_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"🔍 Found {len(image_files)} images.")

with torch.no_grad():
    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        image_id = filename.split('_')[0]
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 224, 224)

        _, latent_vector = model(img_tensor)  # shape: (1, 128)
        latent_vector = latent_vector.cpu().numpy().squeeze()  # shape: (128,)

        results.append([image_id] + latent_vector.tolist())

# ========================
# 写入 CSV 文件
# ========================
header = ["eid"] + [f"latent_{i+1}" for i in range(128)]

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

print(f"✅ 潜在变量已保存到：{output_csv}")
