from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os

class AfhqCat(VisionDataset):
    """Some Information about AfhqCat"""
    def __init__(self, root, transform=None):
        super(AfhqCat, self).__init__(root, transform=transform)
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        img, target = None, -1
        img = Image.open(os.path.join(self.root, f'cat_{index}.jpg')).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        # sum of images in data/afhq/train/cat
        return len([file for file in os.listdir(self.root) if file.endswith('.jpg')])