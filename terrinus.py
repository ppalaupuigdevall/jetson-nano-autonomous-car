import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt


class TerrinusDataset(data.Dataset):
    
    def __init__(self, imgs_dir, masks_dir, transforms, num_classes = 4):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.imgs = os.listdir(imgs_dir)
        self.masks = os.listdir(masks_dir)
        self.C = num_classes
        self.transform = transforms
    
    
    def fetch_mask(self, mask, color_code=[0,64,128,255], mask_tensor=False):
        mask = np.array(mask)
        if(mask_tensor):
            # Load mask of W*H*C (Regression)
            mask_np = np.zeros((np.shape(mask)[0], np.shape(mask)[1], self.C), dtype=np.float32)
            
            for i, c in enumerate(color_code):
                pstns = np.where(mask == c)
                mask_np[pstns[0], pstns[1], i] = 1
        else:
            # Load mask of W*H containing the class number in each position (Cross Entropy)
            mask_np = np.zeros((np.shape(mask)[0],np.shape(mask)[1]), dtype= np.float64)
            for i,c in enumerate(color_code):
                pstns = np.where(mask == c)
                mask_np[pstns[0], pstns[1]] = i
        return mask_np


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.imgs_dir,self.imgs[index]))
        mask = self.fetch_mask(Image.open(os.path.join(self.masks_dir, self.masks[index])))
        img = self.transform(img)
        mask = torch.from_numpy(mask)
        return img, mask
    
    
    def __len__(self):
        return len(self.imgs)