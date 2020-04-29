from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import resize
from albumentations import Rotate, RandomCrop, RandomBrightnessContrast, RandomBrightness, RandomContrast, RandomSizedCrop, RandomGamma, GaussNoise, Compose, Flip, HorizontalFlip
from albumentations.pytorch import ToTensor

class RetinaDataset(Dataset):

    def __init__(self, file_path, transforms=None):
        self.image_list = os.listdir(os.getcwd() + file_path)
        self.root_dir = os.getcwd()+file_path
        if transforms != None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_list[idx])

        image = cv2.imread(img_name)

        image = {'image':cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)}
        image = {'image':resize(image['image'], [768, 768], anti_aliasing=True)}
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        if self.transforms:
            image = self.transforms(image = image['image'])
        return torch.unsqueeze(image['image'], 0)
