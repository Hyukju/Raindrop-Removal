import os
import cv2 
import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ModelDataset(Dataset):
    def __init__(self, data_dir, dataA_dir = 'data', dataB_dir='gt', use_transform=False):
        self.data_dir = data_dir
        self.use_transform = use_transform
        self.transform =  transforms.Compose([
                                            RandomCrop(shape=(256, 256)),
                                            RandomFlip(),
                                            Normalization(mean=0.5, std=0.5),
                                            ToTensor(),
                                            ])
        self.dataA_dir = dataA_dir
        self.dataB_dir = dataB_dir

        lst_dataA = os.listdir(os.path.join(self.data_dir,self.dataA_dir))
        lst_dataA = [f for f in lst_dataA if f.endswith('jpg') | f.endswith('png')]
        lst_dataA.sort()

        lst_dataB = os.listdir(os.path.join(self.data_dir,self.dataB_dir))
        lst_dataB = [f for f in lst_dataB if f.endswith('jpg') | f.endswith('png')]
        lst_dataB.sort()

        self.lst_dataA = lst_dataA
        self.lst_dataB = lst_dataB

    def __len__(self):
        return len(self.lst_dataA)

    def __getitem__(self, index):

        imgA = cv2.imread(os.path.join(self.data_dir, self.dataA_dir, self.lst_dataA[index]))
        imgB = cv2.imread(os.path.join(self.data_dir, self.dataB_dir, self.lst_dataB[index]))
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        if imgA.dtype == np.uint8:
            imgA = imgA / 255.0  
        if imgB.dtype == np.uint8:
            imgB = imgB / 255.0      

        grayA = cv2.cvtColor(imgA.astype('float32'), cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(imgB.astype('float32'), cv2.COLOR_RGB2GRAY)

        mask = (grayA - grayB) * 0.5 + 0.5
     
        data = {'input':imgA, 'label':imgB, 'mask':mask}

        if self.use_transform:
            data = self.transform(data)

        return data

class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for k, v in data.items():
            data[k] = cv2.resize(v, dsize=self.shape)

        return data

class ToTensor():
    def __call__(self, data):
        for k, v in data.items():
            if v.ndim == 2:
                v = v[:,:,np.newaxis]
            data[k] = torch.from_numpy(v.transpose((2,0,1)).astype('float32'))

        return data

class Normalization():
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std 

    def __call__(self, data):
        
        for k, v in data.items():
            data[k] = (v - self.mean) / self.std

        return data


class RandomFlip():
    def __call__(self, data):

        flag_lr = False 
        flag_ud = False 

        if np.random.rand() > 0.5: flag_lr = True
        if np.random.rand() > 0.5: flag_ud = True

        for k, v in data.items():
            if flag_lr: v = np.fliplr(v)
            if flag_ud: v = np.flipud(v)
            data[k] = v

        return data

class RandomCrop():
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        input  = data['input']
        h, w = input.shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for k, v in data.items():
            data[k] = v[id_y, id_x]

        return data
    