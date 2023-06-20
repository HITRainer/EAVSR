import numpy as np
import os
from os.path import join
import cv2
import torch
import math
from tqdm import tqdm
import lpips
import argparse
from skimage.metrics import structural_similarity as ssim


def calc_psnr_np(sr, hr, range=255.):
    # shave = 2
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
    # diff = diff[shave:-shave, shave:-shave, :]
    total_mse = np.power(diff, 2)
    total_psnr = -10 * math.log10(total_mse.mean())

    return total_psnr



def lpips_norm(img):
    img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
    img = img / (255. / 2.) - 1
    return torch.Tensor(img).to(device)


def calc_lpips(x_mask_out, x_canon, loss_fn_alex_1, loss_fn_alex_0=None):
    lpips_mask_out = lpips_norm(x_mask_out)
    lpips_canon = lpips_norm(x_canon)
    # LPIPS_0 = loss_fn_alex_0(lpips_mask_out, lpips_canon)
    LPIPS_1 = loss_fn_alex_1(lpips_mask_out, lpips_canon)
    return LPIPS_1.detach().cpu()  # , LPIPS_1.detach().cpu()



def calc_metrics(out, ref, s):
    total_psnr = calc_psnr_np(out, ref)
    total_ssim = ssim(out, ref, win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
    total_lpips = calc_lpips(out, ref, loss_fn_alex_1)

    return [total_psnr, total_ssim, total_lpips]


def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--name', '-n', help='test name')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--device', default="1")
    parser.add_argument('--load_iter', default="200")
    parser.add_argument('--full_res', type=str2bool, default=False)
    args = parser.parse_args()

    print(args)
    s = 4
    rootlist = [args.dataroot]
    for root in rootlist:
        if os.path.isdir(root):
            root = root
            break

    args.device = "cuda:" + args.device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss_fn_alex_1 = lpips.LPIPS(net='alex', version='0.1').to(device)

    files = [
        '/hdd2/wrh/v2/ckpt/' + args.name + '/'
    ]
    if args.full_res:
        ori_target = '%s/HR_test' % root
    else:
        ori_target = '%s/HR_test' % root

    for file in files:
        if args.full_res:
            log_dir = '%s/log_full_%s.txt' % (file, args.load_iter)
        else:
            log_dir = '%s/log_patch_%s.txt' % (file, args.load_iter)
        f = open(log_dir, 'w')

        metrics_final_result = []
        for folder in sorted(os.listdir(ori_target)):
            names = []
            for file_name in os.listdir(os.path.join(ori_target, folder)):
                #if file_name[-5] == '0' or file_name[-5] == '5':
                #   continue
                names.append(folder + '_' + file_name)
            if not names:
                continue
            names = sorted(names)
            f.write('\n=============%s=============\n' % (folder))
            print('\n=============%s=============\n' % (folder))

            ori_metrics = np.zeros([len(names), 3])
            i = 0
            for name in tqdm(names):
                if args.full_res:
                    print(file + 'sr_full_' + args.load_iter + '/%s/%s' % (name[:3], name[-9:]))
                    pre_out = cv2.imread(file + 'sr_full_' + args.load_iter + '/%s/%s' % (name[:3], name[-9:]))[...,
                              ::-1]
                else:
                    pre_out = cv2.imread(file + 'sr_patch_' + args.load_iter + '/%s/%s' % (name[:3], name[-9:]))[...,
                              ::-1]
                out = pre_out
                pre_ref = cv2.imread(ori_target + '/%s/%s' % (name[:3], name[-9:]))[..., ::-1]
                ref = pre_ref

                ori_metrics[i] = calc_metrics(out, ref, s)
                f.write('name: %s, \n total_psnr: %.2f, \n total_SSIM: %.4f, \n total_LPIPS: %.3f, \n' \
                        % (name, ori_metrics[i][0], ori_metrics[i][1], ori_metrics[i][2]))
                print('name: %s, \n total_psnr: %.2f, \n total_SSIM: %.4f, \n total_LPIPS: %.3f, \n' \
                      % (name, ori_metrics[i][0], ori_metrics[i][1], ori_metrics[i][2]))
                i = i + 1
            metrics_mean = np.mean(ori_metrics, axis=0)
            metrics_final_result.append(metrics_mean)
            f.write('\n folder: %s ======  \
                     \n total_psnr: %.2f, \
                     \n total_SSIM: %.4f, \
                     \n total_LPIPS: %.3f \t \n' \
                    % (folder, metrics_mean[0], metrics_mean[1], metrics_mean[2]))
            print('\n folder: %s ======  \
                     \n total_psnr: %.2f, \
                     \n total_SSIM: %.4f, \
                     \n total_LPIPS: %.3f \t \n' \
                  % (folder, metrics_mean[0], metrics_mean[1], metrics_mean[2]))
        result = np.mean(metrics_final_result, axis=0)
        f.write('\n Final: ======  \
                \n total_psnr: %.2f, \
                \n total_SSIM: %.4f, \
                \n total_LPIPS: %.3f \t \n' \
                % (result[0], result[1], result[2]))
        print('\n Final: ======  \
                \n total_psnr: %.2f, \
                \n total_SSIM: %.4f, \
                \n total_LPIPS: %.3f \t \n' \
              % (result[0], result[1], result[2]))

        f.flush()
        f.close()