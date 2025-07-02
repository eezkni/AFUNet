#-*- coding:utf-8 -*-  
import numpy as np
import os, glob
import cv2
import math
import imageio
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from numpy.random import uniform
import logging
import logging.handlers
imageio.plugins.freeimage.download()

  
class Exposure(object):
    def __init__(self, stops=0.0, gamma=1.0):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img * (2 ** self.stops), 0, 1) ** self.gamma


class PercentileExposure(object):
    def __init__(self, gamma=2.0, low_perc=10, high_perc=90, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            low_perc = uniform(0, 15)
            high_perc = uniform(85, 100)
        self.gamma = gamma
        self.low_perc = low_perc
        self.high_perc = high_perc

    def __call__(self, x):
        low, high = np.percentile(x, (self.low_perc, self.high_perc))
        return map_range(np.clip(x, low, high)) ** (1 / self.gamma)


class BaseTMO(object):
    def __call__(self, img):
        return self.op.process(img)


class Reinhard(BaseTMO):
    def __init__(
            self,
            intensity=-1.0,
            light_adapt=0.8,
            color_adapt=0.0,
            gamma=2.0,
            randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Mantiuk(BaseTMO):
    def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            scale = uniform(0.65, 0.85)

        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )


class Drago(BaseTMO):
    def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            bias = uniform(0.7, 0.9)

        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )


class Durand(BaseTMO):
    def __init__(
            self,
            contrast=3,
            saturation=1.0,
            sigma_space=8,
            sigma_color=0.4,
            gamma=2.0,
            randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            contrast = uniform(3.5)
        self.op = cv2.createTonemapDurand(
            contrast=contrast,
            saturation=saturation,
            sigma_space=sigma_space,
            sigma_color=sigma_color,
            gamma=gamma,
        )


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


TRAIN_TMO_DICT = {
    'exposure': PercentileExposure,
    'reinhard': Reinhard,
    'mantiuk': Mantiuk,
    'drago': Drago,
    'durand': Durand,
}


def random_tone_map(x):
    x = map_range(x)
    tmos = list(TRAIN_TMO_DICT.keys())
    choice = 1
    tmo = TRAIN_TMO_DICT[tmos[choice]](randomize=False)
    return map_range(tmo(x))


def map_range_LDR(x, mode='norm'):
    if mode == 'norm':
        out = (x / 255.0).astype('float32')
        return out * 2. - 1.
    elif mode == 'to_uint8':
        # x = (x + 1.) / 2.
        out = (x * 255).astype('uint8')
        return out 


def freeze_model(model, not_freeze_list, keep_step=None):
    
    for (name, param) in model.named_parameters():
        if name not in not_freeze_list:
            param.requires_grad = False
        else:
            pass

    freezed_num, pass_num = 0, 0
    for (name, param) in model.named_parameters():
        if param.requires_grad == False:
            freezed_num += 1
        else:
            pass_num += 1
    print('\n Total {} params, miss {} \n'.format(freezed_num + pass_num, pass_num))

    return model


def cal_mask_ratio(args, epoch):
    if epoch <= args.phase1_epochs:
        args.mask_ratio = 0.
    else:
        if args.is_curriculum and epoch<=(args.phase1_epochs+50):
            args.mask_ratio = (epoch-args.phase1_epochs) * 0.018
        elif args.is_curriculum and epoch>(args.phase1_epochs+50):
            args.mask_ratio = 1.8 - (epoch-args.phase1_epochs) * 0.018
        else:
            args.mask_ratio = 0.75


def reinhard_tonemapping(hdr_image, gamma=2.2, a=0.18):
    ldr_image = hdr_image / (hdr_image + 1.0)
    ldr_image = torch.pow(ldr_image, 1.0 / gamma)
    ldr_image = ldr_image * (1.0 + ldr_image * a) / (1.0 + ldr_image)
    return ldr_image


def calculate_mask_map(args, reference_img, masks):
    B, C, H, W = reference_img.shape
    mask_map = torch.zeros((B, H, W, 1), dtype=torch.float32, device=reference_img.device)

    for i in range(mask_map.shape[0]):
        mask = masks[i][0]
        mask_ids = torch.unique(mask)
        for mid in mask_ids:
            mask_ratio = min(args.mask_ratio_dict[mid.item()], 1.0)
            m = (mask == mid)
            target_mask_count = torch.round(m.sum() * mask_ratio).int()
            unmasked_indices = (m!=0).nonzero(as_tuple=False)
            random_indices = unmasked_indices[torch.randperm(unmasked_indices.size(0))[:target_mask_count.item()]]
            m[random_indices[:,0], random_indices[:, 1]] = 0
            m = m[..., None]
            mask_map[i] = torch.logical_or(mask_map[i], m.float())
    return mask_map.permute(0,3,1,2)


def calculate_loss_map(args, reference_img, masks):
    B, C, H, W = reference_img.shape
    loss_map = torch.zeros((B, H, W, 1), dtype=torch.float32, device=reference_img.device)
    for i in range(loss_map.shape[0]):
        mask = masks[i][0]
        mask_ids = torch.unique(mask)
        for mid in mask_ids:
            loss_ratio = args.loss_ratio_dict[mid.item()]
            m = (mask == mid)
            loss_map[i][m] = loss_ratio
    return loss_map.permute(0,3,1,2)


def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))


def list_all_files_sorted_with_prefix(folder_name, extension="", prefix=""):
    return sorted(glob.glob(os.path.join(folder_name, prefix + "*" + extension)))


def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))


def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)


def read_label(file_path, file_name):
    label = cv2.imread(os.path.join(file_path, file_name), cv2.IMREAD_UNCHANGED)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    return label


def read_hdr(hdr_imgs, mean, std,  mulog=True, resize=224):
    h, w = hdr_imgs.shape[2:]
    if h == resize:
        imgs = hdr_imgs
    else:
        imgs = F.interpolate(hdr_imgs, size=resize, mode='bilinear', align_corners=False)
    if mulog:
        imgs = range_compressor(imgs)
    else:
        imgs = reinhard_tonemapping(imgs)
    imgs = (imgs - mean) / std
    return imgs


def read_ldr(ldr_imgs, mean, std, resize=224):
    imgs = F.interpolate(ldr_imgs, size=resize, mode='bilinear', align_corners=False)
    imgs = (imgs - mean) / std
    return imgs


def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)


def range_compressor(hdr_img, mu=5000):
    if isinstance(hdr_img, np.ndarray):
        return (np.log(1 + mu * hdr_img)) / math.log(1 + mu)
    elif isinstance(hdr_img, torch.Tensor):
        return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)
    else:
        raise NotImplementedError('range compressor for [%s] is not found' % type(hdr_img))


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data,   0.0)
    print("=== Initialize network with [{}] ===".format(init_type))
    net.apply(init_func)


def calculate_mask_psnr(pred, gt, masks, mask_id_list):
    C, H, W = pred.shape
    d = (pred - gt) ** 2
    psnr_dict = {}
    for mid in mask_id_list:
        mask = (masks == mid)
        mse = np.sum(d[:, mask]) / (C * np.sum(mask))
        psnr_dict[mid] = 10 * np.log10(1 / mse)
    return psnr_dict


def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def save_hdr(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return radiance_writer(path, image)


def random_crop(ldr_images, label, masks, patch_size):
    _, H, W, _ = ldr_images.shape 
    
    h = random.randint(0, max(0, H - patch_size))
    w = random.randint(0, max(0, W - patch_size))
    
    ldr_images = ldr_images[:, h:h+patch_size, w:w+patch_size, :]
    label = label[h:h+patch_size, w:w+patch_size, :]
    masks = masks[h:h+patch_size, w:w+patch_size, :]
    
    return ldr_images, label, masks


def random_crop_no_mask(ldr_images, label, patch_size):
    _, H, W, _ = ldr_images.shape 
    
    h = random.randint(0, max(0, H - patch_size))
    w = random.randint(0, max(0, W - patch_size))
    
    ldr_images = ldr_images[:, h:h+patch_size, w:w+patch_size, :]
    label = label[h:h+patch_size, w:w+patch_size, :]
    
    return ldr_images, label


def resize(ldr_images, label, patch_size, clip_size=0):
    _, w, _ = label.shape
    if clip_size > 0:
        ldr_images = [ldr[:, clip_size:w-clip_size,:]  for ldr in ldr_images]
        label = label[:, clip_size:w-clip_size, :]
    ldr_images = [cv2.resize(ldr, (patch_size, patch_size)) for ldr in ldr_images]
    label = cv2.resize(label, (patch_size, patch_size))    
    return ldr_images, label


def data_augmentation(ldr_images, label, masks):
    if random.random() > 0.5:
        if random.random() > 0.5:
            ldr_images = [cv2.flip(ldr, 0) for ldr in ldr_images]
            label = cv2.flip(label, 0)
            masks = np.expand_dims(cv2.flip(masks, 0), axis=2)
        else:
            ldr_images = [cv2.flip(ldr, 1) for ldr in ldr_images]
            label = cv2.flip(label, 1)
            masks = np.expand_dims(cv2.flip(masks, 1), axis=2)
    return ldr_images, label, masks


def data_augmentation_no_mask(ldr_images, label):
    if random.random() > 0.5:
        if random.random() > 0.5:
            ldr_images = [cv2.flip(ldr, 0) for ldr in ldr_images]
            label = cv2.flip(label, 0)
        else:# horizontal flip
            ldr_images = [cv2.flip(ldr, 1) for ldr in ldr_images]
            label = cv2.flip(label, 1)
    return ldr_images, label


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(log_dir, '{}.log'.format(name))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger_ori(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    return logger


def resizes(ldr_images, label, masks, scale):
    h, w, _ = label.shape
    ldr_images = [cv2.resize(ldr, (w//scale, h//scale)) for ldr in ldr_images]
    label = cv2.resize(label, (w//scale, h//scale))  
    masks = cv2.resize(masks, (w//scale, h//scale))  
    masks = np.expand_dims(masks, axis=-1)
    return ldr_images, label, masks


def resizes_no_mask(ldr_images, label, scale):
    h, w, _ = label.shape
    ldr_images = [cv2.resize(ldr, (w//scale, h//scale)) for ldr in ldr_images]
    label = cv2.resize(label, (w//scale, h//scale))  
    return ldr_images, label