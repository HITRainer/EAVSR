import argparse
import os
import re
from util import util
import torch
import models
import data
import time

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')

inf = float('inf')

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # data parameters
        parser.add_argument('--dataroot', type=str, default='')
        parser.add_argument('--dataset_name', type=str, default=['eth'], nargs='+')
        parser.add_argument('--max_dataset_size', type=int, default=inf)
        parser.add_argument('--scale', type=int, default=2, help='Super-resolution scale.')
        parser.add_argument('--mode', default='RGB', choices=['RGB', 'L', 'Y'],
                help='Currently, only RGB mode is supported.')
        parser.add_argument('--imlib', default='cv2', choices=['cv2', 'pillow'],
                help='Keep using cv2 unless encountered with problems.')
        parser.add_argument('--preload', type=str2bool, default=True,
                help='Load all images into memory for efficiency.')
        parser.add_argument('--multi_imreader', type=str2bool, default=True,
                help='Use multiple cores/threads to load images, will be very '
                     'fast when the images are loaded into cache.')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--patch_size', type=int, default=48)
        parser.add_argument('--lr_mode', default='lr', choices=['lr', 'sr'],
                help='lr: take the lr image directly as input. '
                     'sr: upsample the lr image via bicubic at first.')
        parser.add_argument('--shuffle', type=str2bool, default=True)
        parser.add_argument('-j', '--num_dataloader', default=4, type=int)
        parser.add_argument('--drop_last', type=str2bool, default=True)
        parser.add_argument('--n_frame', type=int, default=5, help='Number of frames processed at once.')
        parser.add_argument('--n_flow', type=int, default=5)
        parser.add_argument('--n_seq', type=int, default=50, help='Totally frames of a video.')

        # device parameters
        parser.add_argument('--gpu_ids', type=str, default='all',
                help='Separate the GPU ids by `,`, using all GPUs by default. '
                     'eg, `--gpu_ids 0`, `--gpu_ids 2,3`, `--gpu_ids -1`(CPU)')
        parser.add_argument('--checkpoints_dir', type=str, default='./ckpt')
        parser.add_argument('-v', '--verbose', type=str2bool, default=True)
        parser.add_argument('--suffix', default='', type=str)

        # model parameters
        parser.add_argument('--name', type=str, required=True,
                help='Name of the folder to save models and logs.')
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--load_path', type=str, default='',
                help='Will load pre-trained model if load_path is set')
        parser.add_argument('--load_iter', type=int, default=[0], nargs='+',
                help='Load parameters if > 0 and load_path is not set. '
                     'Set the value of `last_epoch`')
        parser.add_argument('--n_groups', type=int, default=0)
        parser.add_argument('--n_resblocks', type=int, default=16)
        parser.add_argument('--n_feats', type=int, default=64)
        parser.add_argument('--res_scale', type=float, default=1)
        parser.add_argument('--block_mode', type=str, default='CRC')
        parser.add_argument('--side_ca', type=str2bool, default=False,
                help='If True, put Channel Attention module alongside the '
                     'convolution layers in the residual blocks.')
        parser.add_argument('--predict', type=str2bool, default=True)

        # AdaDSR parameters
        parser.add_argument('--depth', type=float, nargs='+', default=[0])
        parser.add_argument('--sparse_conv', type=str2bool, default=False,\
                help='Replace convolution layers in main body with sparse conv')
        parser.add_argument('--channel_attention', type=str, default='none',
                choices=['none', '0', '1', 'ca'])
        parser.add_argument('--constrain', type=str, default='none',
                choices=['none', 'soft', 'hard'],
                help='none: no constrain on adapter output; '
                     'soft: constrain with depth loss; '
                     'hard: rescale the depth map to a desired average.')
        parser.add_argument('--chop', type=str2bool, default=False)
        parser.add_argument('--alignnet_coord', type=str2bool, default=True)
        parser.add_argument('--ispnet_coord', type=str2bool, default=True)
        # adapter parameters
        parser.add_argument('--nc_adapter', type=int, default=0,
                help='0: no adapter, n: output n channels')
        parser.add_argument('--with_depth', type=str2bool, default=True,
                help='whether adapter take desired depth as input')
        parser.add_argument('--adapter_layers', type=int, default=5)
        parser.add_argument('--adapter_reduction', type=int, default=2)
        parser.add_argument('--adapter_pos', type=int, default=0)
        parser.add_argument('--adapter_bound', type=int, default=None)
        parser.add_argument('--multi_adapter', type=str2bool, default=False)

        # training parameters
        parser.add_argument('--init_type', type=str, default='default',
                choices=['default', 'normal', 'xavier',
                         'kaiming', 'orthogonal', 'uniform'],
                help='`default` means using PyTorch default init functions.')
        parser.add_argument('--init_gain', type=float, default=0.02)
        parser.add_argument('--loss', type=str, default='L1',
                help='choose from [L1, MSE, SSIM, VGG, PSNR]')
        parser.add_argument('--optimizer', type=str, default='Adam',
                choices=['Adam', 'SGD', 'RMSprop'])
        parser.add_argument('--niter', type=int, default=1000)
        parser.add_argument('--niter_decay', type=int, default=0)
        parser.add_argument('--lr_policy', type=str, default='cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=200)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--npost', type=int, default=350, help='The epoch enabled for post-processing operations.')
        parser.add_argument('--full_res', type=str2bool, default=False)

        # Optimizer
        parser.add_argument('--load_optimizers', type=str2bool, default=False,
                help='Loading optimizer parameters for continuing training.')
        parser.add_argument('--weight_decay', type=float, default=0)
        # Adam
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        # SGD & RMSprop
        parser.add_argument('--momentum', type=float, default=0)
        # RMSprop
        parser.add_argument('--alpha', type=float, default=0.99)

        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--test_every', type=int, default=1000)
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--calc_psnr', type=str2bool, default=False)
        parser.add_argument('--save_imgs', type=str2bool, default=False)

        parser.add_argument('--FLOPs_only', type=str2bool, default=False)
        parser.add_argument('--matlab', type=str2bool, default=False)

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are difined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=
                         argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt'
                % ('train' if self.isTrain else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.serial_batches = not opt.shuffle

        if self.isTrain and (opt.load_iter != [0] or opt.load_path != '') \
                and not opt.load_optimizers:
            util.prompt('You are loading a checkpoint and continuing training, '
                        'and no optimizer parameters are loaded. Please make '
                        'sure that the hyper parameters are correctly set.', 80)
            time.sleep(3)

        if opt.mode == 'RGB':
            opt.input_nc = opt.output_nc = 3
        else: # mode = 'L' or 'Y'
            opt.input_nc = opt.output_nc = 1
        opt.model = opt.model.lower()
        opt.name = opt.name.lower()

        scale_patch = {2: 96, 3: 144, 4: 192}
        if opt.patch_size is None:
            opt.patch_size = scale_patch[opt.scale]

        if opt.name.startswith(opt.checkpoints_dir):
            opt.name = opt.name.replace(opt.checkpoints_dir+'/', '')
            if opt.name.endswith('/'):
                opt.name = opt.name[:-1]

        if len(opt.dataset_name) == 1:
            opt.dataset_name = opt.dataset_name[0]
    
        if len(opt.load_iter) == 1:
            opt.load_iter = opt.load_iter[0]

        # process opt.suffix
        if opt.suffix != '':
            suffix = ('_' + opt.suffix.format(**vars(opt)))
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        cuda_device_count = torch.cuda.device_count()
        if opt.gpu_ids == 'all':
            # GT 710 (3.5), GT 610 (2.1)
            gpu_ids = [i for i in range(cuda_device_count)]
        else:
            p = re.compile('[^-0-9]+')
            gpu_ids = [int(i) for i in re.split(p, opt.gpu_ids) if int(i) >= 0]
        opt.gpu_ids = [i for i in gpu_ids \
                       if torch.cuda.get_device_capability(i) >= (4,0)]

        if len(opt.gpu_ids) == 0 and len(gpu_ids) > 0:
            opt.gpu_ids = gpu_ids
            util.prompt('You\'re using GPUs with computing capability < 4')
        elif len(opt.gpu_ids) != len(gpu_ids):
            util.prompt('GPUs(computing capability < 4) have been disabled')

        if len(opt.gpu_ids) > 0:
            assert torch.cuda.is_available(), 'No cuda available !!!'
            torch.cuda.set_device(opt.gpu_ids[0])
            print('The GPUs you are using:')
            for gpu_id in opt.gpu_ids:
                print(' %2d *%s* with capability %d.%d' % (
                        gpu_id,
                        torch.cuda.get_device_name(gpu_id),
                        *torch.cuda.get_device_capability(gpu_id)))
        else:
            util.prompt('You are using CPU mode')

        self.opt = opt
        return self.opt