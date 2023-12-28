import os
import pathlib
import random
from random import shuffle

import cv2
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = pathlib.Path(image_path).name
        GT_path = os.path.join(self.GT_paths, filename)

        image = Image.open(image_path).convert("I")
        GT = Image.open(GT_path).convert("L")

        aspect_ratio = image.size[1] / image.size[0]

        image = np.array(image, dtype=np.float32)
        GT = np.array(GT, dtype=np.float32)
        GT[GT == 255.0] = 1.0

        t = A.Compose([
            A.Resize(height=256, width=256,interpolation=cv2.INTER_CUBIC),
            # A.LongestMaxSize(max_size=256),
            A.Normalize(0.456, 0.224),
            ToTensorV2(),
        ])
        aug = t(image=image, mask=GT)
        image = aug["image"]
        GT = aug["mask"]
        return image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
