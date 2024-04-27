from torch.utils.data import Dataset
import os
from PIL import Image


class CrabDataset(Dataset):
    def __init__(self, root_dir, input_subdir, target_subdir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, input_subdir))
        self.input_subdir = input_subdir
        self.target_subdir = target_subdir

    def __len__(self):
        return len(self.images)

    def get_variance(self):
        import numpy as np
        data_variance = np.var(self.images / 255.0)
        return data_variance

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.input_subdir, self.images[idx])
        image = Image.open(img_name)
        target_name = os.path.join(self.root_dir, self.target_subdir, self.images[idx])
        target = Image.open(target_name)

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target
        

