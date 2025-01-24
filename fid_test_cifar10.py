import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os

def load_stats_if_npz(path):
    if path.endswith('.npz') and os.path.exists(path):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
        return m, s
    return None, None

def calculate_mean_cov(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

class InceptionV3FID(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        # 最後のpoolまでを使用
        self.blocks = nn.Sequential(*list(inception.children())[:-1])
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            return self.blocks(x).view(x.size(0), -1)

def compute_fid(dataloader_real=None, dataloader_fake=None, stats_real_path=None, stats_fake_path=None, device='cpu'):
    """
    dataloader_real と dataloader_fake が与えられれば、そこから特徴を抽出
    stats_real_path と stats_fake_path が .npz へのパスなら、そこから mu, sigma を読み込む
    どちらか一方だけ npz ファイルを使う場合は組み合わせてOK
    """
    model = InceptionV3FID().to(device).eval()

    if stats_real_path:
        mu1, sigma1 = load_stats_if_npz(stats_real_path)
    else:
        mu1, sigma1 = None, None

    if stats_fake_path:
        mu2, sigma2 = load_stats_if_npz(stats_fake_path)
    else:
        mu2, sigma2 = None, None

    # データローダから計算
    if dataloader_real and (mu1 is None or sigma1 is None):
        real_features = []
        for imgs in dataloader_real:
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            real_features.append(feats)
        real_features = np.concatenate(real_features, axis=0)
        mu1, sigma1 = calculate_mean_cov(real_features)

    if dataloader_fake and (mu2 is None or sigma2 is None):
        fake_features = []
        for imgs in dataloader_fake:
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            fake_features.append(feats)
        fake_features = np.concatenate(fake_features, axis=0)
        mu2, sigma2 = calculate_mean_cov(fake_features)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

if __name__ == '__main__':
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor()
    ])

    # ダミーのテンソルでテスト
    dummy_real = [torch.rand(3,299,299) for _ in range(4)]
    dummy_fake = [torch.rand(3,299,299) for _ in range(4)]
    
    loader_real = data.DataLoader(dummy_real, batch_size=2)
    loader_fake = data.DataLoader(dummy_fake, batch_size=2)
    
    # 通常のFID計算
    fid_result = compute_fid(dataloader_real=loader_real, dataloader_fake=loader_fake)
    print("FID with DataLoader:", fid_result)

    # 例のnpzファイルを読み込むパターン（実際にはファイルが存在しないためNoneになる）
    # stats_real_path = "real_stats.npz"
    # stats_fake_path = "fake_stats.npz"
    # fid_npz = compute_fid(stats_real_path=stats_real_path, stats_fake_path=stats_fake_path)
    # print("FID with npz:", fid_npz)
