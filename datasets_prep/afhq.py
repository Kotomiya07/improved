from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os

class AfhqCat(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.image_paths = []
        self.labels = []

        # 画像ファイルのパスを取得
        for filename in os.listdir(root):
            if filename.endswith(".png"):
                self.image_paths.append(os.path.join(root, filename))
                self.labels.append(0)  # 例として全てのラベルを0とする（必要に応じて変更）

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 画像を読み込み
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]

        # 必要に応じて変換を適用
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label