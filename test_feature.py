from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import yaml
from ldm.util import instantiate_from_config
import torchvision
from datasets_prep.dataset import create_dataset

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument('--datadir', default='./data/cifar-10')
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--feature_path", type=str, default="features")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        '--AutoEncoder_config', default='./autoencoder/config/vq-f4.yaml', help='path of config file for AntoEncoder')

    parser.add_argument(
        '--AutoEncoder_ckpt', default='./autoencoder/weight/vq-f4.ckpt', help='path of weight for AntoEncoder')
    

    args = parser.parse_args()

    features_dir = f"{args.feature_path}/{args.dataset}/{args.image_size}_features"
    labels_dir = f"{args.feature_path}/{args.dataset}/{args.image_size}_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    data_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True)
    
    dataset2 = create_dataset(args)
    data_loader2 = DataLoader(dataset2,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True)
    
    config_path = args.AutoEncoder_config
    ckpt_path = args.AutoEncoder_ckpt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset in ['cifar10', 'stl10', 'coco']:

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        AutoEncoder = instantiate_from_config(config['model'])


        checkpoint = torch.load(ckpt_path, map_location=device)
        AutoEncoder.load_state_dict(checkpoint['state_dict'])
        AutoEncoder.eval()
        AutoEncoder.to(device)

    image, label = next(iter(data_loader))
    tmp = image.squeeze(dim=1).to(device)
    label = label.squeeze(dim=1).to(device)
    print(tmp[0].shape, label[0].shape)

    tmp2, label2 = next(iter(data_loader2))
    tmp2 = tmp2.to(device)
    label2 = label2.to(device)
    real = tmp2.clone()

    with torch.no_grad():
        sample = AutoEncoder.decode(tmp)
        tmp2 = AutoEncoder.encode(tmp2)
        sample2 = AutoEncoder.decode(tmp2)
    real = (torch.clamp(real, -1, 1) + 1) / 2  # 0-1
    sample = (torch.clamp(sample, -1, 1) + 1) / 2  # 0-1
    sample2 = (torch.clamp(sample2, -1, 1) + 1) / 2  # 0-1
    torchvision.utils.save_image(
        real, os.path.join("./", 'real_data.png'))
    torchvision.utils.save_image(
        sample, os.path.join("./", 'sample_data.png'))
    torchvision.utils.save_image(
        sample2, os.path.join("./", 'sample_data2.png'))