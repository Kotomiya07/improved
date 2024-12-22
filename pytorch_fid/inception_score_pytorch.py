import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from PIL import Image
import numpy as np
from scipy.stats import entropy

# GPUが利用可能かどうか確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_inception_score(imgs, batch_size=32, splits=10):
    """
    Inception Scoreを計算する関数

    Args:
        imgs (list of PIL.Image.Image): 画像のリスト
        batch_size (int): バッチサイズ
        splits (int): スプリット数

    Returns:
        tuple: (Inception Scoreの平均, Inception Scoreの標準偏差)
    """
    N = len(imgs)
    assert N > 0, "画像がありません"

    # Inception-v3モデルのロード（学習済み）
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # データ変換
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_pred(batch):
        batch = torch.stack([transform(img) for img in batch]).to(device)
        with torch.no_grad():
            output = model(batch)
        return torch.softmax(output, dim=1).cpu().numpy()

    # 全ての画像の予測確率を取得
    preds = np.zeros((N, 1000))
    n_batches = N // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        preds[start:end] = get_pred(imgs[start:end])
    if n_batches * batch_size < N:
        start = n_batches * batch_size
        preds[start:] = get_pred(imgs[start:])

    # Inception Scoreの計算
    scores = []
    for i in range(splits):
        part = preds[(i * N // splits):((i + 1) * N // splits)]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.sum(kl, 1)
        scores.append(np.exp(np.mean(kl)))

    return np.mean(scores), np.std(scores)

def load_images_from_path(image_path):
    """
    指定されたパスから画像を読み込む関数

    Args:
        image_path (str): 画像ファイルまたはディレクトリのパス

    Returns:
        list of PIL.Image.Image: 読み込まれた画像のリスト
    """
    import os

    images = []
    if os.path.isfile(image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Error loading image: {image_path} - {e}")
    elif os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    img_path = os.path.join(image_path, filename)
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image: {img_path} - {e}")
    else:
        print(f"指定されたパスはファイルまたはディレクトリではありません: {image_path}")
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Inception Score for images.')
    parser.add_argument('--image_path', type=str, help='Path to the image file or directory containing images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images.')
    parser.add_argument('--splits', type=int, default=10, help='Number of splits for Inception Score calculation.')
    args = parser.parse_args()

    image_path = args.image_path
    batch_size = args.batch_size
    splits = args.splits

    # 画像をロード
    images = load_images_from_path(image_path)

    if not images:
        print(f"指定されたパスに有効な画像が見つかりませんでした: {image_path}")
        exit()

    # Inception Scoreを計算
    mean_score, std_score = calculate_inception_score(images, batch_size, splits)

    print(f"Inception Score: {mean_score:.4f} (±{std_score:.4f})")