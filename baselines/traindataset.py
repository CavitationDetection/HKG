import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale, Lambda
import os

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 1.0),
    transforms.RandomVerticalFlip(p = 1.0),
    transforms.RandomRotation(degrees = [180, 180]),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])



class TrainImgDataset(torch.utils.data.Dataset):

    def __init__(self, opt, transform = None):
        self.opt = opt
        self.root = opt.train_root
        self.label = opt.train_label_path
        self.img_files = os.listdir(self.root)
        img_id = np.array([int(file.split('.')[0]) for file in self.img_files])
        self.labels = np.array(pd.read_csv(self.label, header=0)['label'])[img_id]
        self.transform = transform


    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.img_files[idx])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img , self.labels[idx]

    
    def __len__(self):
        return len(self.img_files)

