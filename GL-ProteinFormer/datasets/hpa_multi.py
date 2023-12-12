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



class HpaMultiDataset(data.Dataset):
    def __init__(self, split='train', root=None, annotation=None, using = 'multi_only'):
        self.split = split
        img_list = np.loadtxt(annotation, dtype=str)
        sublocations=['golgiapparatus','mitochondrion','vesicles','endoplasmicreticulum'
             ,'nucleolus','nucleus','cytoskeleton','lysosome','cytoplasm']
        
        image_root = osp.join(root, split)
        self.image_list = []
        self.label_list = []

        self.n_classes = 9

        for image_name in img_list:
            image_path = osp.join(image_root, image_name)
            
#             print('image_name', image_name)
#             sublocation = image_name.split('_')[2][:-1]
            sublocation = image_name.split('.')[:-1][0].split('_')[2][:-1].lower()
            sublocation = sublocation.split('+')
#             print('sublocation', sublocation)
#             label = None
            label = np.zeros(len(sublocations))
            for j in range(len(sublocation)):
                for i in range(len(sublocations)):
                    if (sublocation[j] == sublocations[i]):
#                         label.append(i)
                        label[i] = 1
#             print('label', label)
            if using == 'multi_only':
                num_using = 1
            elif using == 'all':
                num_using = 0
                
            if label.sum() > num_using :
                self.image_list.append(image_path)
                self.label_list.append(label)

    # def get_label_encoded(self, labels):
    #     print('labels', labels)
    #     # label = np.zeros(shape=(self.n_classes), dtype=np.float32)
    #     # for i in range(self.n_classes):
    #     #     label[i] = 1 if i == labels else 0
    #     # # print('label', label)
    #     return labels

    def __getitem__(self, index):
        if self.split == 'train':
#             print('train data')
            image = Image.open(self.image_list[index])

            preprocess = T.Compose([T.RandomHorizontalFlip(p=0.5),
                                    T.RandomVerticalFlip(p=0.5),
                                    T.RandomRotation(degrees=(-180, 180))])
            image = preprocess(image)

            # image = image.resize((256,256))
            image = image.resize((224,224))
            image = image.convert('RGB')
            image = np.asarray(image, dtype='float32').transpose(2, 0, 1)
            image = image * 1.0 / 255
            label = self.label_list[index]
            # label = self.get_label_encoded(label)
        elif self.split == 'test':
#             print('test data')
            image = Image.open(self.image_list[index])
            # image = image.resize((256,256))
            image = image.resize((224,224))
            image = image.convert('RGB')
            image = np.asarray(image, dtype='float32').transpose(2, 0, 1)
            image = image * 1.0 / 255
            label = self.label_list[index]
            # label = self.get_label_encoded(label)
        else:
            print('split error !!!!!!!!!!!!!!!!!!!!!!!!!')
        
        return {"input": image,
                "target": label}
        
        
    def __len__(self):
        return len(self.image_list)