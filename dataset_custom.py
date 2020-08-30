import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob

class CustomDataset(data.Dataset):
    """
    Assumes only one object per frame.
    Only mask for first frame.

    DATASET
    ├── 00000.jpg  # first frame
    ├── 00000.jpg  # first frame
    ├── 00001.jpg
    └── mask.png   # mask for frame
    """

    def __init__(self, root):
        self.root = root
        self.K = 2
        self.frames = sorted([f for f in os.listdir(root) if f.split('.')[-1] == "jpg"])
        self.mask = "mask.png"

    def __len__(self):
        return 1 # Number of videos, not number of frames

    def __getitem__(self, index):
        Fs = [self.get_image(f) for f in self.frames]
        Fs = np.array(Fs).transpose((3, 0, 1, 2))

        M = self.get_image(self.mask, mask=True)
        Ms = np.expand_dims(M, axis=1)

        num_objects = torch.LongTensor([int(1)])
        info = {"name": self.root, "num_frames": len(self.frames)}
        return Fs, Ms, num_objects, info

    def get_image(self, f, mask=False):
        im = Image.open(os.path.join(self.root, f))
        if mask:
            im = (np.array(im.convert("L")) == 255).astype(np.uint8)
            # im = Image.fromarray(im)
            # im = np.array(im.convert("P")).astype(np.uint8)
            im = np.array([im == k for k in range(self.K)])
            print(im.shape)
            return im.astype(np.uint8)

        # Normalize to [0, 1]
        else:
            im = np.array(im.convert("RGB"))
            return (im / 255).astype(np.float32)


class CustomDatasetMultiMask(data.Dataset):
    """
    Assumes only one object per frame.
    Only mask for first frame.

    DATASET
    ├── 00000.jpg   
    ├── 00001.jpg   
    ├── 00002.jpg
    ├── xxxxx.png   # mask
    └── xxxxx.png   # mask
    """

    def __init__(self, root):
        self.root = root
        self.K = 2
        self.frames = sorted([f for f in os.listdir(root) if f.split('.')[-1] == "jpg"])
        self.masks = sorted([f for f in os.listdir(root) if f.split('.')[-1] == "png"])
        self.mask_ids = [self.frames.index(m.replace('png', 'jpg')) for m in self.masks]

    def __len__(self):
        return 1 # Number of videos, not number of frames

    def __getitem__(self, index):
        Fs = [self.get_image(f) for f in self.frames]
        # print('a', np.array(Fs).shape)
        Fs = np.array(Fs).transpose((3, 0, 1, 2))

        Ms = [self.get_image(f, mask=True) for f in self.masks]
        # print('b', np.array(Ms).shape)
        Ms = np.array(Ms).transpose((1, 0, 2, 3))

        num_objects = torch.LongTensor([int(1)])
        info = {"name": self.root, "num_frames": len(self.frames), "mask_ids": self.mask_ids}
        return Fs, Ms, num_objects, info

    def get_image(self, f, mask=False):
        im = Image.open(os.path.join(self.root, f))

        # Convert into palette and split into K channels
        if mask:
            # TODO: changes this line depending on how your mask is encoded
            im = (np.array(im) == 255).astype(np.uint8)
            im = Image.fromarray(im)
            im = np.array(im.convert("P")).astype(np.uint8)
            im = np.array([im == k for k in range(self.K)])
            return im.astype(np.uint8)

        # Normalize to [0, 1]
        else:
            im = np.array(im.convert("RGB"))
            return (im / 255).astype(np.float32)


if __name__ == '__main__':
    pass
