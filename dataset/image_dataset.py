#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *

class Image_Dataset(Dataset):
    def __init__(self, ldr_path, label_path, exposure_path, masks_path, patch_size, hdr_name, stride_size=None):
        self.ldr_images = read_images(ldr_path)
        self.label = read_label(label_path, hdr_name)
        self.masks = np.load(masks_path, allow_pickle=True)
        if stride_size==None:
            self.stride_size = patch_size // 4
        else:
            self.stride_size = stride_size
        self.ldr_patches, self.hdr_patches, self.mask_patches = self.get_ordered_patches(patch_size, self.stride_size)
        self.expo_times = read_expo_times(exposure_path)
        self.patch_size = patch_size
        self.result = []

    def __getitem__(self, index):
        pre_imgs = [ldr_to_hdr(self.ldr_patches[index][i], self.expo_times[i], 2.2) for i in range(len(self.ldr_patches[index]))] 
        pre_imgs = [np.concatenate((pre_imgs[i], self.ldr_patches[index][i]), 2) for i in range(len(self.ldr_patches[index]))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(self.ldr_patches[index]))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(self.ldr_patches[index]))]

        label = self.hdr_patches[index].astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        masks = self.mask_patches[index].astype(np.float32).transpose(2, 0, 1)
        masks = torch.from_numpy(masks)

        sample = {
            'inputs' : imgs,
            'label' : label,
            'masks' : masks
            }
        return sample

    def get_ordered_patches(self, patch_size, stride_size):
        
        ldr_patch_list = []
        hdr_patch_list = []
        mask_patch_list = []
        h, w, c = self.label.shape
        ldr_num = len(self.ldr_images)
        n_h = h // stride_size + 1 
        n_w = w // stride_size + 1
        tmp_h = n_h * stride_size
        tmp_w = n_w * stride_size
        tmp_ldrs = np.ones((ldr_num, tmp_h, tmp_w, c), dtype=np.float32)
        tmp_hdr = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_mask = np.zeros((tmp_h, tmp_w, 1), dtype=np.float32)
        for i in range(ldr_num):
            tmp_ldrs[i, :h, :w] = self.ldr_images[i]
        tmp_hdr[:h, :w] = self.label
        tmp_mask[:h, :w] = self.masks

        for x in range(n_w):
            for y in range(n_h):
                if x * stride_size + patch_size <= tmp_w and y * stride_size + patch_size <= tmp_h:
                    temp_patch_ldrs = tmp_ldrs[:, y*stride_size:y*stride_size+patch_size, x*stride_size:x*stride_size+patch_size]
                    temp_patch_ldrs = [temp_patch_ldrs[i] for i in range(ldr_num)]
                    ldr_patch_list.append(temp_patch_ldrs)
                    temp_patch_hdr = tmp_hdr[y*stride_size:y*stride_size+patch_size, x*stride_size:x*stride_size+patch_size]
                    hdr_patch_list.append(temp_patch_hdr)
                    temp_patch_mask = tmp_mask[y*stride_size:y*stride_size+patch_size, x*stride_size:x*stride_size+patch_size]
                    mask_patch_list.append(temp_patch_mask)
        
        #assert len(ldr_patch_list) == n_h * n_w
        return ldr_patch_list, hdr_patch_list, mask_patch_list

    def __len__(self):
        return len(self.ldr_patches)
    
    def get_mask_map(self):
        return self.masks
    
    def get_folder_name(self):  
        return self.folder_name

    def rebuild_result(self):
        h, w, c = self.label.shape
        n_h = h // self.stride_size + 1 
        n_w = w // self.stride_size + 1
        tmp_h = n_h * self.stride_size
        tmp_w = n_w * self.stride_size
        pred = np.zeros((c, tmp_h, tmp_w), dtype=np.float32)
        vis = np.zeros((tmp_h,tmp_w),dtype=np.float32)
        cnt = 0
        for x in range(n_w):
            for y in range(n_h):
                if x * self.stride_size + self.patch_size <= tmp_w and y * self.stride_size + self.patch_size <= tmp_h:
                    pred[:, y*self.stride_size:y*self.stride_size+self.patch_size, x*self.stride_size:x*self.stride_size+self.patch_size] *= vis[y*self.stride_size:y*self.stride_size+self.patch_size, x*self.stride_size:x*self.stride_size+self.patch_size]
                    vis[y*self.stride_size:y*self.stride_size+self.patch_size, x*self.stride_size:x*self.stride_size+self.patch_size] += 1
                    pred[:, y*self.stride_size:y*self.stride_size+self.patch_size, x*self.stride_size:x*self.stride_size+self.patch_size] += self.result[cnt]
                    cnt+=1
                    pred[:, y*self.stride_size:y*self.stride_size+self.patch_size, x*self.stride_size:x*self.stride_size+self.patch_size] /= vis[y*self.stride_size:y*self.stride_size+self.patch_size, x*self.stride_size:x*self.stride_size+self.patch_size]
        return pred[:, :h, :w], self.label.transpose(2, 0, 1)
    
    def update_result(self, tensor):
        self.result.append(tensor)
