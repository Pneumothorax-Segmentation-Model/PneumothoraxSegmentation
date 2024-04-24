import os
import pandas as pd
from torch.utils.data import Dataset
from utils.data import generate_data
import numpy as np
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torch


class PneumothoraxDataset(Dataset):
    def __init__(
        self,
        root,
        anotations_file,
        img_dir,
        mask_dir,
        dcm_dir=None,
        transform=None,
        generate=False,
    ):
        super().__init__()
        self.img_labels = pd.read_csv(os.path.join(root, anotations_file))

        if generate:
            dcm_dir = os.path.join(root, dcm_dir)
            mask_dir = os.path.join(root, mask_dir)
            img_dir = os.path.join(root, img_dir)
            generate_data(self.img_labels, root, dcm_dir, img_dir, mask_dir)

        self.img_dir = os.path.join(root, img_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.transform = transform

    def __len__(self):
        # return len(self.img_labels)
        # temporarily set to decrease training time
        return 10

    def __getitem__(self, index):
        img_id = self.img_labels.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_id + ".png")
        img = read_image(img_path, ImageReadMode.GRAY).float()
        label = self.img_labels.iloc[index, 1]
        if label == " -1":
            mask = torch.zeros((1, 1024, 1024)).float()
            return img, mask
        else:
            mask_path = os.path.join(self.mask_dir, img_id + "_mask.png")
            mask = read_image(mask_path, ImageReadMode.GRAY).float()
            mask = mask / 255.0
            return img, mask
