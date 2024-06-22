'''
Description: our dataset
Author: Yu Sha
Date: 2022-04-25 13:47:23
LastEditors: Yu Sha
LastEditTime: 2023-10-26 11:49:49
'''
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import pickle


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

columns_to_select = ['non','cavitation','choked','constant','incipient']

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, opt, transform = None, word_embedding_name = None):
        """
        Args:
            opt: An options object containing configuration.
            transform: A list of image transformations.
            word_embedding_name: Path to word embedding data.
        """
        self.opt = opt
        self.root = opt.test_root
        self.label = opt.test_label_path
        self.transform = transform
        self.img_files = os.listdir(self.root)
        self.img_id_list = np.array([int(file.split('.')[0]) for file in self.img_files])
        self.labels = np.array(pd.read_csv(self.label, header=0)[columns_to_select])[self.img_id_list]

        with open(word_embedding_name, 'rb') as f:
            self.word_embedding = pickle.load(f)
        self.word_embedding_name = word_embedding_name

    def load_and_preprocess_image(self, item):
        img = Image.open(os.path.join(self.root, self.img_files[item])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __getitem__(self, item):
        img = self.load_and_preprocess_image(item).type(torch.float32)
        return (img, self.img_id_list[item], self.word_embedding), self.labels[item]
        
    def __len__(self):
        return len(self.img_files)
