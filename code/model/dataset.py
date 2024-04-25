import os
from PIL import Image
# from pathlib import Path
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    """
         自定义数据集，同时加载图片和文本
    """

    def __init__(self, image_dir, text_path, image_transform=None):
        super(MyDataset, self).__init__()
        self.image_dir = image_dir
        self.text_path = text_path
        # if not self.image_dir.is_dir():
        #     raise RuntimeError(f'Invalid directory "{image_dir}"')
        # if not self.text_path.is_file():
        #     raise RuntimeError(f'Invalid file "{text_path}"')

        self.image_transform = image_transform
        with open(text_path, 'r')  as f:
            self.text_list = f.readlines()
        # self.image_list = sorted(f for f in self.image_dir.iterdir() if f.is_file())
        self.image_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        first, _ = self.image_list[idx].split('.')[0].split('_')
        first = int(first)
        text = self.text_list[first-1][:-1]
        
        return image, text