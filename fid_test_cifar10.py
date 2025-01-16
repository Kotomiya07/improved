#%%
import torch
import torchvision
from torch.nn.functional import interpolate
from scipy import linalg
import numpy as np
from torchvision.models import inception_v3

import torchvision.transforms as transforms
import torch.nn as nn
import os
import subprocess

def calculate_fid(real_features, fake_features):
    # 特徴量の平均と共分散行列を計算
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # 2つの分布間のFIDスコアを計算
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def main():
    # データの前処理
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10データセットをロード
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 特徴抽出用のInceptionV3モデルを準備
    # inception = inception_v3(pretrained=True)
    # inception.fc = nn.Identity()  # 最後の全結合層を除去
    # inception.eval()
    # inception = inception.to(device)

    modes = ['linear', 'bilinear', 'bicubic', 'trilinear']
    mode = modes[2]
    os.makedirs(f"data/cifar-10/reconstract_images_{mode}", exist_ok=True)
    for i, (images, _) in enumerate(trainloader):
        # 32x32 -> 16x16 -> 32x32
        compressed = interpolate(images, size=(16, 16), 
                                mode=f'{mode}', align_corners=False)
        reconstructed = interpolate(compressed, size=(32, 32), 
                                    mode=f'{mode}', align_corners=False)
        torchvision.utils.save_image(reconstructed, f"data/cifar-10/reconstract_images_{mode}/cifar10_{i}.png")
    
    # CIFAR-10を画像で保存
    # os.makedirs("data/cifar-10/images", exist_ok=True)
    # for i, (images, _) in enumerate(trainloader):
    #     torchvision.utils.save_image(images, f"data/cifar-10/images/cifar10_{i}.png")

    # FIDスコアを計算
    #fid_score = calculate_fid(real_features, fake_features)
    real_img_stats = 'data/cifar-10/cifar10-stats.npz'
    save_dir = f'data/cifar-10/reconstract_images_{mode}'
    
    fid_score = subprocess.run(["python", "-m", "pytorch_fid", f"{real_img_stats}", f"{save_dir}", "--device", f"{device}"], stdout=subprocess.PIPE)
    print(f"FID Score: {fid_score.stdout.decode('utf-8')}")

#%%
if __name__ == "__main__":
    main()
# %%

