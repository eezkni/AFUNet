# -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.training_dataset import Training_Dataset, Validing_Dataset, Testing_Dataset
from dataset.test_dataset import Test_Dataset
from loss.loss import Loss
from models.AFUNet import AFUNet
from utils.utils import * 
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from lr_scheduler.mylr import MyLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_ssim

def pad_image(image, patch_size, stride_size):
    _, _, h, w = image.size()
    pad_h = (stride_size - (h-patch_size) % stride_size) % stride_size
    pad_w = (stride_size - (w-patch_size) % stride_size) % stride_size
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode='reflect')
    return padded_image

def get_args():
    parser = argparse.ArgumentParser(description='All Settings',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', type=str, default='logdir_AFUNet',
                        help='target log directory')
    parser.add_argument("--dataset_dir", type=str, default='../data/Kal',
                        help='dataset directory')
    parser.add_argument('--train_path', type=str, default='Training',
                        help='train path(default: Training)')
    parser.add_argument('--test_path', type=str, default='Test/Test-set',
                        help='test path(default: Test)')
    parser.add_argument('--exposure_file_name', type=str, default='exposure.txt',
                        help='exposure file name')
    parser.add_argument('--ldr_prefix', type=str, default='',
                        help='ldr tif prefix string')
    parser.add_argument('--ldr_folder_name', type=str, default=None,
                        help='ldr folder name')
    parser.add_argument('--label_file_name', type=str, default='HDRImg.hdr',
                        help='label file name')
    
    # Training and Test Settings
    parser.add_argument('--train_patch_size', type=int, default=128,
                        help='patch size for training (default: 128)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='patch size for test (default: 256)')
    parser.add_argument('--repeat', type=int, default=100,
                        help='number of repeat for training dataset (default: 100)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data for training (default: 8)')
    parser.add_argument('--test_num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data for test (default: 1)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')  # 1
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 550)')
    parser.add_argument('--phase1_epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train without mask (default: 550)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='training batch size (default: 16)')  # Batch <<<========
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--resume', type=str, default='',  # Resume <<<========
                        help='load model from a .pth file (default: None)')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--cache_choice', type=int, default = 2,
                        help='cache for dataloader(0: none, 1: bin, 2: in_memory)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--lr_decay', action='store_true', default=True,
                        help='learning rate decay or not')
    parser.add_argument('--le_lambda', type=float, default=0.005, metavar='N',
                        help='weight of the local-enhanced loss(default: 0.005)')

    # Other Settings
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--is_freeze', action='store_true', default=False,
                        help='freeze partial parameters or not')
    return parser.parse_args()

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader, ncols=80)):
        # dataloader
        batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
        batch_ldrs = torch.cat(batch_ldrs, dim=1)
        label = batch_data['label'].to(device)
        pred = model(batch_ldrs)

        # loss
        loss, loss_dict = criterion(pred, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logger_train.info('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'L1 Loss: {:.6f}\t'
                  'Local-Enhanced Loss: {:.6f}'.format(
                epoch,
                batch_idx * args.batch_size,
                len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item(),
                loss_dict['loss_recon'].item(),
                loss_dict['loss_le'].item(),
            ))
            
            tb_writer.add_scalar('train/loss', loss.item(), batch_idx+(epoch-1)*len(train_loader.dataset))
            tb_writer.add_scalar('train/loss_recon', loss_dict['loss_recon'].item(), batch_idx+(epoch-1)*len(train_loader.dataset))
            tb_writer.add_scalar('train/loss_le', loss_dict['loss_le'].item(), batch_idx+(epoch-1)*len(train_loader.dataset))

def test_single_img(args, model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False)
    with torch.no_grad():
        for batch_data in dataloader:
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            output = model(batch_ldrs)
            img_dataset.update_result(torch.squeeze(output.detach().cpu()).numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    return pred, label

def test(args, model, device, optimizer, lr_scheduler, epoch, test_loader, ckpt_dir):
    model.eval()
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, ncols=80)):
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            label = batch_data['label'].to(device)

            padded_image = pad_image(batch_ldrs, args.train_patch_size, args.train_patch_size)
            pred_img = model(padded_image)

            _, _, orig_h, orig_w = label.size()
            pred_img = pred_img[:, :, :orig_h, :orig_w]

            mse_l =  F.mse_loss(label,pred_img)
            scene_psnr_l = (20 * torch.log10(1.0 / torch.sqrt(mse_l)))
            scene_ssim_l = pytorch_ssim.ssim(label, pred_img)

            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            mse_mu =  F.mse_loss(label_mu, pred_img_mu)
            scene_psnr_mu = (20 * torch.log10(1.0 / torch.sqrt(mse_mu)))
            scene_ssim_mu = pytorch_ssim.ssim(label_mu, pred_img_mu)

            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu) 

    if best_metric['psnr_l']['value'] < psnr_l.avg:
        best_metric['psnr_l']['value'] = psnr_l.avg
        best_metric['psnr_l']['epoch'] = epoch
    if best_metric['psnr_mu']['value'] < psnr_mu.avg:
        best_metric['psnr_mu']['value'] = psnr_mu.avg
        best_metric['psnr_mu']['epoch'] = epoch
    if best_metric['ssim_l']['value'] < ssim_l.avg:
        best_metric['ssim_l']['value'] = ssim_l.avg
        best_metric['ssim_l']['epoch'] = epoch
    if best_metric['ssim_mu']['value'] < ssim_mu.avg:
        best_metric['ssim_mu']['value'] = ssim_mu.avg
        best_metric['ssim_mu']['epoch'] = epoch

    logger_train.info('Epoch:' + str(epoch))
    logger_train.info('Test set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}\n'.format(
        psnr_l.avg,
        psnr_mu.avg,
        ssim_l.avg,
        ssim_mu.avg
        ))
    logger_valid.info('==Best==\tPSNR_l: {:.4f}/epoch: {}\t PSNR_mu: {:.4f}/epoch: {} \t SSIM_l: {:.4f}/epoch: {}\t SSIM_mu: {:.4f}/epoch: {}'.format(
        best_metric['psnr_l']['value'], best_metric['psnr_l']['epoch'],
        best_metric['psnr_mu']['value'], best_metric['psnr_mu']['epoch'],
        best_metric['ssim_l']['value'], best_metric['ssim_l']['epoch'],
        best_metric['ssim_mu']['value'], best_metric['ssim_mu']['epoch']
    ))

    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(save_dict, os.path.join(ckpt_dir, 'epoch_{:d}.pth'.format(epoch)))
    tb_writer.add_scalar('test/psnr_l', psnr_l.avg, epoch)
    tb_writer.add_scalar('test/psnr_mu', psnr_mu.avg, epoch)
    tb_writer.add_scalar('test/ssim_l', ssim_l.avg, epoch)
    tb_writer.add_scalar('test/ssim_mu', ssim_mu.avg, epoch)

def main():
    print('===> Init settings')
    args = get_args()
    
    if args.seed is not None:
        set_random_seed(args.seed)
    
    logdir = args.logdir
    tensorboard_dir_curve = os.path.join(logdir, 'tensorboard','curve')
    tensorboard_dir_figure = os.path.join(logdir, 'tensorboard','figure')
    ckpt_dir = os.path.join(logdir, 'ckpt')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(tensorboard_dir_curve):
        os.makedirs(tensorboard_dir_curve)
    if not os.path.exists(tensorboard_dir_figure):
        os.makedirs(tensorboard_dir_figure)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global logger_train
    logger_train = get_logger('train', logdir)
    global logger_valid
    logger_valid = get_logger('valid', logdir)
    global tb_writer
    tb_writer = SummaryWriter(os.path.join(tensorboard_dir_curve))
    global tb_figure
    tb_figure = SummaryWriter(os.path.join(tensorboard_dir_figure))

    args_dict = vars(args)
    for key, value in args_dict.items():
        logger_train.info(f'{key}: {value}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('===> Loading datasets')
    if args.cache_choice == 0:
        cache = 'none'
        print('===> No cache')
    elif args.cache_choice == 1:
        cache = 'bin'
        print('===> Cache bin')
    elif args.cache_choice == 2:
        cache = 'in_memory'
        print('===> Cache in_memory')

    train_dataset = Training_Dataset(root_dir=args.dataset_dir, 
                                     patch_size=args.train_patch_size, 
                                     repeat=args.repeat, cache=cache, 
                                     train_path=args.train_path, 
                                     exposure_file_name=args.exposure_file_name, 
                                     ldr_folder_name=args.ldr_folder_name, 
                                     label_file_name=args.label_file_name,
                                     ldr_prefix=args.ldr_prefix)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, 
                              pin_memory=True)  
    trainset_size = len(train_loader.dataset)

    test_dataset = Testing_Dataset(root_dir=args.dataset_dir, 
                                    patch_size=args.patch_size, 
                                    repeat=1, cache=cache, 
                                    train_path=args.test_path, 
                                    exposure_file_name=args.exposure_file_name, 
                                    ldr_folder_name=args.ldr_folder_name, 
                                    label_file_name=args.label_file_name,
                                    ldr_prefix=args.ldr_prefix)
    
    test_loader = DataLoader(test_dataset, batch_size=1, 
                              shuffle=False, num_workers=1, 
                              pin_memory=True) 
    
    testset_size = len(test_loader.dataset)

    print('===> Training dataset size: {},Testing dataset size: {}.'.format(trainset_size, testset_size))

    upscale = 4
    window_size = 8
    height = (128 // upscale // window_size + 1) * window_size
    width = (128 // upscale // window_size + 1) * window_size
    model = AFUNet(img_size=(height, width), in_chans=18, window_size=window_size, img_range=1., drop_path_rate=0.1, depths=[5, 5, 5, 5], embed_dim=72, num_heads=[4, 4, 4, 4], mlp_ratio=2)
    
    # init
    if args.init_weights:
        init_weights(model, init_type='normal', gain=0.02)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)# 1e-8
    
    # lr_scheduler
    if args.lr_decay:
        lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch = args.phase1_epochs, eta_min=5e-6)

    model.to(device)
    model = nn.DataParallel(model)

    # load checkpoint
    if args.resume and os.path.isfile(args.resume):
        if args.is_freeze:
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 1
            model.load_state_dict(checkpoint['state_dict'])
            model = freeze_model(model=model, not_freeze_list=['module.conv_first.0.weight', 'module.conv_first.0.bias'])
            # optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                         lr=args.lr, betas=(0.9, 0.999), eps=1e-8)# 1e-8
            if args.lr_decay:
                lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch = args.phase1_epochs, 
                                    eta_min=5e-6)
            print("===> Start fine-tuning.")
        else:
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
    else:
        print("===> No checkpoint is founded.")

    # model complexity
    from ptflops import get_model_complexity_info
    with torch.no_grad():
        flops, params = get_model_complexity_info(model, (18, 128, 128), as_strings=True, print_per_layer_stat=False, verbose=False)
        logger_train.info(f'### flops: {flops}, params: {params}.')
        print('## Flops: ', flops, ', Params: ', params)

    # loss  
    criterion = Loss(le_lambda = args.le_lambda).to(device)
    # metrics
    global best_metric
    best_metric = {'psnr_l': {'value': 0., 'epoch': 0},
                   'psnr_mu': {'value': 0., 'epoch': 0},
                   'ssim_l': {'value': 0., 'epoch': 0},
                   'ssim_mu': {'value': 0., 'epoch': 0}}

    for epoch in range(args.start_epoch, args.epochs + 1):
        logger_train.info(f'===> Epoch: {epoch}/{args.epochs}')
        print(f'===> Epoch: {epoch}/{args.epochs}')
        
        for param_group in optimizer.param_groups:
            logger_train.info("Learning rate is: [{:1.7f}] ==".format(param_group['lr']))
            tb_writer.add_scalar('train/lr', param_group['lr'], epoch)
            print("Learning rate is: [{:1.7f}] ==".format(param_group['lr']))
        
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        if args.lr_decay:
            lr_scheduler.step()
        
        print(f"==> start test of epoch {epoch}.")
        test(args, model, device, optimizer, lr_scheduler, epoch, test_loader, ckpt_dir)


if __name__ == '__main__':
    main()

