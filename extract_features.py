import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import os
from diffusers.models import AutoencoderKL
from datasets_prep.dataset import create_dataset
from ldm.util import instantiate_from_config
import yaml
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device("cuda")

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(f'{args.features_path}/{args.dataset}/{args.image_size}_features', exist_ok=True)
    os.makedirs(f'{args.features_path}/{args.dataset}/{args.image_size}_labels', exist_ok=True)

    # Load VAE:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    config_path = args.AutoEncoder_config
    ckpt_path = args.AutoEncoder_ckpt
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    vae = instantiate_from_config(config['model'])
    text_encoder = FrozenCLIPEmbedder(device="cpu")


    checkpoint = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()
    vae.to(device)


    # Setup data:
    dataset = create_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x, y in loader:
        x = x.to(device)
        if args.dataset == "cifar10":
            y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x)
            if args.dataset == "coco":
                y = text_encoder(y[0])

        x = x.sample().detach().cpu().numpy()    # (1, 4, 32, 32)
        np.save(f'{args.features_path}/{args.dataset}/{args.image_size}_features/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.features_path}/{args.dataset}/{args.image_size}_labels/{train_steps}.npy', y)

        train_steps += 1
        if train_steps % 100 == 0:
            print(train_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cifar10","celeba","lsun","coco","imagenet"], required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--features_path", type=str, default="features")
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        '--AutoEncoder_config', default='./autoencoder/config/kl-f2.yaml', help='path of config file for AntoEncoder')
    parser.add_argument(
        '--AutoEncoder_ckpt', default='./autoencoder/weight/kl-f2.ckpt', help='path of weight for AntoEncoder')

    args = parser.parse_args()
    main(args)
