from torch.utils.data import Dataset
import numpy as np
import random
import torch
import cv2
import os
import glob
import random

class myDataset(Dataset):
    def __init__(self,path=''):
        super().__init__()
        self.pth = sorted(glob.glob( os.path.join(path,'*_r2r_iput.png')))
        self.num = len(self.pth)
        self.index = list(range(self.num))
        # random.shuffle(self.index)

    def __len__(self):
        return (self.num)

    def __getitem__(self, index):
        img_path = self.pth[self.index[index]]
        
        r2r_input= cv2.imread(img_path).astype(np.float32).transpose(2,0,1,)/255.
        r2r_output= cv2.imread(img_path.replace('r2r_iput','r2r_output')).astype(np.float32).transpose(2,0,1,)/255.
        
        return torch.from_numpy(r2r_input),torch.from_numpy(r2r_output)
