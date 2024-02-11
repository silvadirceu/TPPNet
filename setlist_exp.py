import os, sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *
import glob
from torch.utils.data import Dataset
import torch, torch.utils
import random
import bisect
import torchvision
import PIL
import deepdish as dd 

def cut_data(data, out_length=None):
    
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
            
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
        
    return data

def cut_data_front(data, out_length):
    
    if out_length is not None:
        if data.shape[0] > out_length:
            offset = 0
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
            
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
        
    return data

def shorter(feature, mean_size=2):
    length, height  = feature.shape
    new_f = np.zeros((int(length/mean_size),height),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_f[i,:] = feature[i*mean_size:(i+1)*mean_size,:].mean(axis=0)
    return new_f

class CQTLoader(Dataset):
    
    def __init__(self, in_dir, mode='dir', out_length=None, listfile=None, ext=".npy"):
        self.indir = in_dir
        self.mode=mode
        
        if mode == 'listfile': 
            with open(listfile, 'r') as fp:
                self.file_list = [line.rstrip() for line in fp]
        else:
            self.file_list = glob.glob(os.path.join(self.indir,"**/*"+ext), recursive=True)
            
        self.out_length = out_length
        
    def __getitem__(self, index):
        
        transform_test = transforms.Compose([
            lambda x : x.T,
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        
        filename = self.file_list[index].strip()
        parts = filename.split('/')
        set_id = parts[-2]
        version_id = parts.split('.')[0]
        
        data = np.load(filename) # from 12xN to Nx12

        data = transform_test(data)

        return data, set_id, version_id
    
    def __len__(self):
        return len(self.file_list)

def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def predict(model, dataloader, out_dir=None):
    
    model.eval()
    labels, features, versions = [], None, []
    
    for ii, (data, label, version) in enumerate(dataloader):
        
        print(f"Computing: {ii} de {len(dataloader)}")
        
        input_data = data.to(torch.device(device()))

        feature = model(input_data)
        feature = feature.data.cpu().numpy()
        
        if out_dir is not None:
            features = norm(feature)
            
            path_dir = os.path.join(out_dir,label)
            os.makedirs(path_dir, exist_ok=True)
            out_file = os.path.join(path_dir,version+".h5")
            dd.save(out_file,(features, label, version))
        else:
            features = np.concatenate((features, feature), axis=0)
            labels.append(label)
            versions.append(version)

    if out_dir is None:
        features = norm(features)
        return features, labels, versions


if __name__=="__main__":
    
    import models

    opt.model = ''
    opt.load_model_path = 'check_points/best.pth'
    tpp_model = getattr(models, opt.model)()
    tpp_model.load(opt.load_model_path)

    tpp_model.to(torch.device(device()))

    tpp_out = '/mnt/dev/dirceusilva/dados/Cover/setlist_all/setlist_ecad/features/universe_tpp'
    cqt_in = '/mnt/dev/dirceusilva/dados/Cover/setlist_all/setlist_ecad/features/universe_cqt'

    setlist_data = CQTLoader(in_dir=cqt_in, ext=".npy")
    data_loader = DataLoader(setlist_data, 1, shuffle=False, num_workers=1)
    predict(model=tpp_model, dataloader=data_loader, out_dir=tpp_out)


