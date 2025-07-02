from dataset.training_dataset import Testing_Dataset
from models.AFUNet import AFUNet
from utils.utils import *
from skimage.metrics import peak_signal_noise_ratio
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='../data/Kal',
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--pretrained_model', type=str, default='./pretrain_model/Kalantari.pth')
parser.add_argument('--test_best', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./results/")
parser.add_argument('--model_arch', type=int, default=0)
parser.add_argument('--test_path', type=str, default='Test/Test-set',
                    help='test path(default: Test)')
parser.add_argument('--exposure_file_name', type=str, default='exposure.txt',
                    help='exposure file name(default: exposure.txt)')
parser.add_argument('--ldr_prefix', type=str, default='',
                        help='ldr tif prefix string')
parser.add_argument('--ldr_folder_name', type=str, default=None,
                    help='ldr folder name(default: None)')
parser.add_argument('--label_file_name', type=str, default='HDRImg.hdr',
                    help='label file name(default: HDRImg.hdr)')
parser.add_argument('--cache_choice', type=int, default = 2,
                    help='cache for dataloader(0: none, 1: bin, 2: in_memory)')
parser.add_argument('--test_num_workers', type=int, default=1, metavar='N',
                    help='number of workers to fetch data for test (default: 1)')

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

def pad_image(image, patch_size, stride_size):
    _, _, h, w = image.size()
    pad_h = (stride_size - (h-patch_size) % stride_size) % stride_size
    pad_w = (stride_size - (w-patch_size) % stride_size) % stride_size
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode='reflect')
    return padded_image

def main():
    # Settings
    args = parser.parse_args()

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)
    print(args.patch_size)

    # cuda and devices
    use_cuda = torch.cuda.is_available()
    
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("Device: ", device)

    upscale = 4
    window_size = 8
    height = (128 // upscale // window_size + 1) * window_size
    width = (128 // upscale // window_size + 1) * window_size

    # model architecture
    model_dict = {
        0: AFUNet(img_size=(height, width), in_chans=18, window_size=window_size, 
                  img_range=1., drop_path_rate=0.1, depths=[5, 5, 5, 5], embed_dim=72,
                  num_heads=[4, 4, 4, 4], mlp_ratio=2),
    }
    print(f"Selected model: {args.model_arch}")
    model = model_dict[args.model_arch].to(device)
    model = nn.DataParallel(model, device_ids = [0])
    model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
    model.eval()
    print(f">>>>>>> Model Loaded: {args.pretrained_model} >>>>>>>>>")
    
    test_dataset = Testing_Dataset(root_dir=args.dataset_dir, 
                                    patch_size=args.patch_size, 
                                    repeat=1, cache = "in_memory", 
                                    train_path=args.test_path, 
                                    exposure_file_name=args.exposure_file_name, 
                                    ldr_folder_name=args.ldr_folder_name, 
                                    label_file_name=args.label_file_name)
    test_loader = DataLoader(test_dataset, batch_size=1, 
                              shuffle=False, num_workers=1, 
                              pin_memory=True) 
    testset_size = len(test_loader.dataset)
    
    print("testset_size: ", testset_size)
    
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            label = batch_data['label'].to(device)
            padded_image = pad_image(batch_ldrs, args.patch_size, args.patch_size)
            pred_img = model(padded_image)
            _, _, orig_h, orig_w = label.size()
            pred_img = pred_img[:, :, :orig_h, :orig_w]
            
            pred_img = torch.squeeze(pred_img.detach().cpu()).numpy().astype(np.float32)
            label = torch.squeeze(label.detach().cpu()).numpy().astype(np.float32)
            print(">>>>>>>>> Testing Scene: {} >>>>>>>>>".format(idx))
            pred_hdr = pred_img.copy()
            pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]

            # psnr-l and psnr-\mu
            scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)

            # ssim-l
            pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
            label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
            scene_ssim_l = calculate_ssim(pred_img, label)
            # ssim-\mu
            pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu)

            print(f' {idx} | 'f'PSNR_mu: {scene_psnr_mu:.4f}  PSNR_l: {scene_psnr_l:.4f} | SSIM_mu: {scene_ssim_mu:.4f}  SSIM_l: {scene_ssim_l:.4f}')

            if args.save_results:
                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                # save results
                # cv2.imwrite(os.path.join(args.save_dir, '00{}_pred.hdr'.format(idx)), pred_hdr)

        print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
        print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
        print(">>>>>>>>> Finish Testing >>>>>>>>>")

if __name__ == '__main__':
    main()