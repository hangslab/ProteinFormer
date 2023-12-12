import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

import os
import math
import random
from glob import glob
import os.path as osp


class HpaDataset(data.Dataset):
    def __init__(self, split='train', root=None, annotation=None):
        self.split = split
        img_list = np.loadtxt(annotation, dtype=str)
#         print('img_list', img_list)
#         sublocations=['Golgiapparatus','Mitochondrion','Vesicles','Endoplasmicreticulum'
#              ,'Nucleolus','Nucleus','Cytoskeleton']
        sublocations=['golgiapparatus','mitochondrion','vesicles','endoplasmicreticulum'
             ,'nucleolus','nucleus','cytoskeleton']
        
        image_root = osp.join(root, split)
        self.image_list = []
        self.label_list = []

        bins = [0, 0, 0, 0, 0, 0, 0]

        for image_name in img_list:
            image_path = osp.join(image_root, image_name)
            label = None
#             print('image_name', image_name)
#             sublocation = image_name.split('_')[2][:-1]
            sublocation = image_name.split('.')[:-1][0].split('_')[2][:-1].lower()
#             print('sublocation', sublocation)
            for i in range(len(sublocations)):
                if (sublocation == sublocations[i]):
                    label = i
            
            if label is not None:
                bins[label] = bins[label] + 1
                self.image_list.append(image_path)
                self.label_list.append(label)
        
        print('bins', bins)
        self.bins = bins

    def __getitem__(self, index):
        if self.split == 'train':
#             print('train data')
            image = Image.open(self.image_list[index])

            preprocess = T.Compose([T.RandomHorizontalFlip(p=0.5),
                                    T.RandomVerticalFlip(p=0.5),
                                    T.RandomRotation(degrees=(-180, 180))])
            image = preprocess(image)

            image = image.resize((224,224))
            image = image.convert('RGB')
            image = np.asarray(image, dtype='float32').transpose(2, 0, 1)
            image = image * 1.0 / 255
            label = self.label_list[index]
        elif self.split == 'test':
#             print('test data')
            image = Image.open(self.image_list[index])
            image = image.resize((224,224))
            image = image.convert('RGB')
            image = np.asarray(image, dtype='float32').transpose(2, 0, 1)
            image = image * 1.0 / 255
            label = self.label_list[index]
        else:
            print('split error !!!!!!!!!!!!!!!!!!!!!!!!!')
        
        return image, label
        
        
    def __len__(self):
        return len(self.image_list)