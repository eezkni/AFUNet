#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
from utils.utils import *
from dataset.image_dataset import Image_Dataset


def Test_Dataset(root_dir, patch_size, mode,
                 test_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 mask_npy_name):
    scenes_dir = osp.join(root_dir, test_path)
    scenes_list = sorted(os.listdir(scenes_dir))
    ldr_list = []
    label_list = []
    expo_times_list = []
    mask_npy_list = []
    if mode == 'train':
        stride_size = patch_size
    elif mode == 'test':
        stride_size = None
    for scene in range(len(scenes_list)):
        exposure_file_path = os.path.join(scenes_dir, scenes_list[scene], exposure_file_name)
        mask_npy_path = os.path.join(scenes_dir, scenes_list[scene], mask_npy_name)
        if ldr_folder_name is None:
            ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scenes_list[scene]), '.tif')
        else:
            ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scenes_list[scene], ldr_folder_name), '.tif')
        label_path = os.path.join(scenes_dir, scenes_list[scene])
        expo_times_list += [exposure_file_path]
        mask_npy_list += [mask_npy_path]
        ldr_list += [ldr_file_path]
        label_list += [label_path]

    for ldr_dir, label_dir, expo_times_dir, mask_npy_dir in zip(ldr_list, label_list, expo_times_list, mask_npy_list):
        yield Image_Dataset(ldr_dir, label_dir, expo_times_dir, mask_npy_dir, patch_size, label_file_name, stride_size)