import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import gc
import random
import torch.nn.functional as F
import pandas as pd
import time
from torchvision.transforms import InterpolationMode, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

def norm_vol(data):
    data = data.astype(np.float32)  
    min_val, max_val = np.min(data), np.max(data)
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)  
def normalize_01_into_pm1(x):  
    return x.add(x).add_(-1)
class ISIC(Dataset):
    def __init__(self, data_dir="", 
                 train_flag=True, csv_file=""):
        self.train_flag=train_flag
        self.post_trans = transforms.Compose([normalize_01_into_pm1])
        self.data_dir=data_dir 
            
        self.data = []
        self.load_from_csv(csv_file,train_flag)
        self.data_len = len(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name,label= self.data[index]  
        
        image = self.get_subject(image_name)
        image = norm_vol(image)  

        image = torch.from_numpy(image).float() 
        
        image = image.permute(2, 0, 1)
        
        image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        image=self.post_trans(image)

        
        return image


    def get_subject(self,image_name):
        
        image_name += '.jpg'
        path = os.path.join(self.data_dir, image_name)
        image = np.array(Image.open(path))  

        return image

    def load_from_csv(self, csv_file, train_flag=True):
        
        df_data = pd.read_csv(csv_file)

        
        if train_flag:
            df_data, _ = train_test_split(df_data, train_size=0.9, stratify=df_data['dx'], random_state=42)
        else:
            _, df_data = train_test_split(df_data, test_size=0.1, stratify=df_data['dx'], random_state=42)

        column1 = 'image_id'
        column2 = 'dx'
        self.data = list(zip(df_data[column1], df_data[column2]))

if __name__ == "__main__": 
    ISIC_dataset  =ISIC(train_flag=True)
    ISIC_dataloder = DataLoader(ISIC_dataset, batch_size=64, shuffle=True)
    for x,image in enumerate(ISIC_dataloder):
        print(image.shape)