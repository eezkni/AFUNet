#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *
import pickle


class Training_Dataset(Dataset):

    def __init__(self, root_dir, patch_size, repeat, cache,
                 train_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 ldr_prefix = ""):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.repeat = repeat
        self.cache = cache
        self.label_file_name = label_file_name

        self.scenes_dir = osp.join(root_dir, train_path)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(os.path.join(self.scenes_dir, self.scenes_list[scene]), exposure_file_name)
            if ldr_folder_name is None:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif', ldr_prefix)
            else:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(self.scenes_dir, self.scenes_list[scene], ldr_folder_name), '.tif', ldr_prefix)
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            
            if cache == 'none':
                self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_dir),
                    '_bin_' + os.path.basename(root_dir))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                exposure_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_exposure.pkl')
                if not os.path.exists(exposure_bin_file):
                    with open(exposure_bin_file, 'wb') as f:
                        pickle.dump(read_expo_times(exposure_file_path), f)
                    print('dump', exposure_bin_file)
                ldrs_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_ldr.pkl')
                if not os.path.exists(ldrs_bin_file):
                    with open(ldrs_bin_file, 'wb') as f:
                        pickle.dump(read_images(ldr_file_path), f)
                    print('dump', ldrs_bin_file)
                label_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_label.pkl')
                if not os.path.exists(label_bin_file):
                    with open(label_bin_file, 'wb') as f:
                        pickle.dump(read_label(label_path, label_file_name), f)
                    print('dump', label_bin_file)
                self.image_list.append([exposure_bin_file, ldrs_bin_file, label_bin_file])

            elif cache == 'in_memory':
                # Read exposure times
                expoTimes = read_expo_times(exposure_file_path)
                # Read LDR images
                ldr_images = read_images(ldr_file_path)
                # Read HDR label
                label = read_label(label_path, label_file_name)
                # Read mask
                # masks = np.load(mask_npy_path, allow_pickle=True)
                self.image_list.append([expoTimes, ldr_images, label])

    def __getitem__(self, index):

        # calculate index
        index = index % len(self.scenes_list)

        if self.cache == 'none':
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])

            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            
            # Read HDR label
            label = read_label(self.image_list[index][2], self.label_file_name)
        
        elif self.cache == 'bin':
            with open(self.image_list[index][0], 'rb') as f:
                expoTimes = pickle.load(f)
            with open(self.image_list[index][1], 'rb') as f:
                ldr_images = pickle.load(f)
            with open(self.image_list[index][2], 'rb') as f:
                label = pickle.load(f)
            # with open(self.image_list[index][3], 'rb') as f:
            #     masks = pickle.load(f)

        elif self.cache == 'in_memory':
            expoTimes, ldr_images, label = self.image_list[index]
        
        # Random crop
        ldr_images, label = random_crop_no_mask(ldr_images, label, self.patch_size)

        # data augmentation
        ldr_images, label = data_augmentation_no_mask(ldr_images, label)
        
        # ldr images process
        pre_imgs = [ldr_to_hdr(ldr_images[i], expoTimes[i], 2.2) for i in range(len(ldr_images))]    
        pre_imgs = [np.concatenate((pre_imgs[i], ldr_images[i]), 2) for i in range(len(ldr_images))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(ldr_images))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(ldr_images))]        
        
        # hdr image process
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        # mask process
        # masks = masks.astype(np.float32).transpose(2, 0, 1)
        # masks = torch.from_numpy(masks)
        
        sample = {
            'inputs': imgs, 
            'label': label,
            # 'masks': masks,
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)*self.repeat
    

class Validing_Dataset(Dataset):
    def __init__(self, root_dir, patch_size, repeat, cache,
                 train_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 ldr_prefix = ""):
        self.root_dir = root_dir  # /Kalantari
        self.patch_size = patch_size  # 128
        self.repeat = repeat
        self.cache = cache
        self.label_file_name = label_file_name

        self.scenes_dir = osp.join(root_dir, train_path)  # /Kalantari/Training
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(os.path.join(self.scenes_dir, self.scenes_list[scene]), exposure_file_name)
            # mask_npy_path = os.path.join(os.path.join(self.scenes_dir, self.scenes_list[scene]), mask_npy_name)
            if ldr_folder_name is None:
                ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            else:
                ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene], ldr_folder_name), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            
            if cache == 'none':
                self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_dir),
                    '_bin_valid_' + os.path.basename(root_dir))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                exposure_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_exposure.pkl')
                if not os.path.exists(exposure_bin_file):
                    with open(exposure_bin_file, 'wb') as f:
                        pickle.dump(read_expo_times(exposure_file_path), f)
                    print('dump', exposure_bin_file)
                ldrs_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_ldr.pkl')
                if not os.path.exists(ldrs_bin_file):
                    with open(ldrs_bin_file, 'wb') as f:
                        pickle.dump(read_images(ldr_file_path), f)
                    print('dump', ldrs_bin_file)
                label_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_label.pkl')
                if not os.path.exists(label_bin_file):
                    with open(label_bin_file, 'wb') as f:
                        pickle.dump(read_label(label_path, label_file_name), f)
                    print('dump', label_bin_file)
                # masks_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_mask.pkl')
                # if not os.path.exists(masks_bin_file):
                #     with open(masks_bin_file, 'wb') as f:
                #         pickle.dump(np.load(mask_npy_path, allow_pickle=True), f)
                #     print('dump', masks_bin_file)
                self.image_list.append([exposure_bin_file, ldrs_bin_file, label_bin_file])

            elif cache == 'in_memory':
                # Read exposure times
                expoTimes = read_expo_times(exposure_file_path)
                # Read LDR images
                ldr_images = read_images(ldr_file_path)
                # Read HDR label
                label = read_label(label_path, label_file_name)
                # Read mask
                # masks = np.load(mask_npy_path, allow_pickle=True)
                self.image_list.append([expoTimes, ldr_images, label])

    def __getitem__(self, index):

        # calculate index
        index = index % len(self.scenes_list)

        if self.cache == 'none':
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])

            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            
            # Read HDR label
            label = read_label(self.image_list[index][2], self.label_file_name)

            # Read mask
            # masks = np.load(self.image_list[index][3], allow_pickle=True)
        
        elif self.cache == 'bin':
            with open(self.image_list[index][0], 'rb') as f:
                expoTimes = pickle.load(f)
            with open(self.image_list[index][1], 'rb') as f:
                ldr_images = pickle.load(f)
            with open(self.image_list[index][2], 'rb') as f:
                label = pickle.load(f)
            # with open(self.image_list[index][3], 'rb') as f:
            #     masks = pickle.load(f)

        elif self.cache == 'in_memory':
            expoTimes, ldr_images, label = self.image_list[index]
        
        # print("--------",masks.shape)        
        ldr_images, label = resizes_no_mask(ldr_images, label, 4)  # 3
        # ldr_images, label, masks = resizes(ldr_images, label, masks, 2)
        # print("--------",masks.shape)

        # # data augmentation
        # ldr_images, label, masks = data_augmentation(ldr_images, label, masks)
        
        # ldr images process
        pre_imgs = [ldr_to_hdr(ldr_images[i], expoTimes[i], 2.2) for i in range(len(ldr_images))]    
        pre_imgs = [np.concatenate((pre_imgs[i], ldr_images[i]), 2) for i in range(len(ldr_images))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(ldr_images))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(ldr_images))]        
        
        # hdr image process
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        # mask process
        # masks = masks.astype(np.float32).transpose(2, 0, 1)
        # masks = torch.from_numpy(masks)
        
        sample = {
            'inputs': imgs, 
            'label': label,
            # 'masks': masks,
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)*self.repeat


class Testing_Dataset(Dataset):
    def __init__(self, root_dir, patch_size, repeat, cache,
                 train_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 ldr_prefix = ""):
        self.root_dir = root_dir  # /Kalantari
        self.patch_size = patch_size  # 128
        self.repeat = repeat
        self.cache = cache
        self.label_file_name = label_file_name

        self.scenes_dir = osp.join(root_dir, train_path)  # /Kalantari/Test
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        # self.scenes_list = self.scenes_list[::5]  # <<<<<<<<<<<================= 

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(os.path.join(self.scenes_dir, self.scenes_list[scene]), exposure_file_name)
            # mask_npy_path = os.path.join(os.path.join(self.scenes_dir, self.scenes_list[scene]), mask_npy_name)
            if ldr_folder_name is None:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif', ldr_prefix)
            else:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(self.scenes_dir, self.scenes_list[scene], ldr_folder_name), '.tif',ldr_prefix)
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            
            if cache == 'none':
                self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_dir),
                    '_bin_test_' + os.path.basename(root_dir))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                exposure_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_exposure.pkl')
                if not os.path.exists(exposure_bin_file):
                    with open(exposure_bin_file, 'wb') as f:
                        pickle.dump(read_expo_times(exposure_file_path), f)
                    print('dump', exposure_bin_file)
                ldrs_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_ldr.pkl')
                if not os.path.exists(ldrs_bin_file):
                    with open(ldrs_bin_file, 'wb') as f:
                        pickle.dump(read_images(ldr_file_path), f)
                    print('dump', ldrs_bin_file)
                label_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_label.pkl')
                if not os.path.exists(label_bin_file):
                    with open(label_bin_file, 'wb') as f:
                        pickle.dump(read_label(label_path, label_file_name), f)
                    print('dump', label_bin_file)

            elif cache == 'in_memory':
                # Read exposure times
                expoTimes = read_expo_times(exposure_file_path)
                # Read LDR images
                ldr_images = read_images(ldr_file_path)
                # Read HDR label
                label = read_label(label_path, label_file_name)
                # Read mask
                # masks = np.load(mask_npy_path, allow_pickle=True)
                self.image_list.append([expoTimes, ldr_images, label])

    def __getitem__(self, index):

        # calculate index
        index = index % len(self.scenes_list)

        scence = self.scenes_list[index]
        name = os.path.basename(scence)

        if self.cache == 'none':
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])

            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            
            # Read HDR label
            label = read_label(self.image_list[index][2], self.label_file_name)

            # Read mask
            # masks = np.load(self.image_list[index][3], allow_pickle=True)
        
        elif self.cache == 'bin':
            with open(self.image_list[index][0], 'rb') as f:
                expoTimes = pickle.load(f)
            with open(self.image_list[index][1], 'rb') as f:
                ldr_images = pickle.load(f)
            with open(self.image_list[index][2], 'rb') as f:
                label = pickle.load(f)
            # with open(self.image_list[index][3], 'rb') as f:
            #     masks = pickle.load(f)

        elif self.cache == 'in_memory':
            expoTimes, ldr_images, label = self.image_list[index]
        
        # ldr images process
        pre_imgs = [ldr_to_hdr(ldr_images[i], expoTimes[i], 2.2) for i in range(len(ldr_images))]    
        pre_imgs = [np.concatenate((pre_imgs[i], ldr_images[i]), 2) for i in range(len(ldr_images))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(ldr_images))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(ldr_images))]        
        
        # hdr image process
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        sample = {
            'inputs': imgs, 
            'label': label,
            # 'masks': masks,
            'name': name,
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)*self.repeat


