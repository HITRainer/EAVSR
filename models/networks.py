import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
from util.util import extract_image_patches
import torchvision.ops as ops
import torchvision.models.vgg as vgg



def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,
										step_size=opt.lr_decay_iters,
										gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
												   mode='min',
												   factor=0.2,
												   threshold=0.01,
												   patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
												   T_max=opt.niter,
												   eta_min=0)
	else:
		return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
				or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, b=init_gain)
			else:
				raise NotImplementedError('[%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	if init_type != 'default' and init_type is not None:
		init_weights(net, init_type, init_gain=init_gain)
	return net

'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
	if len(args) == 1:
		args = args[0]
	if isinstance(args, nn.Module):
		return args
	modules = OrderedDict()
	if isinstance(args, OrderedDict):
		for k, v in args.items():
			modules[k] = seq(v)
		return nn.Sequential(modules)
	assert isinstance(args, (list, tuple))
	return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
		 output_padding=0, dilation=1, groups=1, bias=True,
		 padding_mode='zeros', mode='CBR'):
	L = []
	for t in mode:
		if t == 'C':
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=groups,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'X':
			assert in_channels == out_channels
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=in_channels,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels,
										out_channels=out_channels,
										kernel_size=kernel_size,
										stride=stride,
										padding=padding,
										output_padding=output_padding,
										groups=groups,
										bias=bias,
										dilation=dilation,
										padding_mode=padding_mode))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'i':
			L.append(nn.InstanceNorm2d(out_channels))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'S':
			L.append(nn.Sigmoid())
		elif t == 'P':
			L.append(nn.PReLU())
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return seq(*L)


class MeanShift(nn.Conv2d):
	""" is implemented via group conv """
	def __init__(self, rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040),
				 rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1, groups=3)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.ones(3).view(3, 1, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False


def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3,
						  stride=1, padding=1, bias=True, mode='2R'):
	# mode examples: 2, 2R, 2BR, 3, ..., 4BR.
	assert len(mode)<4 and mode[0] in ['2', '3', '4']
	up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size,
			   stride, padding, bias=bias, mode='C'+mode)
	return up1


class PatchSelect(nn.Module):
	def __init__(self,  stride=1):
		super(PatchSelect, self).__init__()
		self.stride = stride             

	def rgb2gray(self, rgb):
		r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
		gray = 0.2989*r + 0.5870*g + 0.1140*b
		return gray

	def forward(self, query, key):
		# query: lr, [N, C, H, W]
		# key:  ref, [N, C, 4*H, 4*W]
		# query = self.rgb2gray(query)
		# key = self.rgb2gray(key)

		shape_query = query.shape
		shape_key = key.shape
		
		P = shape_key[3] - shape_query[3] + 1  # patch number per row
		key = extract_image_patches(key, ksizes=[shape_query[2], shape_query[3]], 
				strides=[self.stride, self.stride], rates=[1, 1], padding='valid')
		# [N, C*H*W, P*P']

		key = key.view(shape_query[0], shape_query[1], shape_query[2] * shape_query[3], -1)
		sorted_key, _ = torch.sort(key, dim=2)

		query = query.view(shape_query[0], shape_query[1], shape_query[2] * shape_query[3], 1)
		sorted_query, _ = torch.sort(query, dim=2)
		y = torch.mean(torch.mean(torch.abs(sorted_key - sorted_query), 2), 1)  # [N, P*P']

		# query = query.view(shape_query[0], shape_query[1] * shape_query[2] * shape_query[3], 1)
		# y = torch.mean(torch.abs(key - query), 1)  # [N, P*P']
		relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True)  # [N, P*P']   
		
		return  hard_indices.view(-1), P # , relavance_maps


class AdaptBlockFeat(nn.Module):
	def __init__(self, opt, inplanes=64, outplanes=64, stride=1, dilation=1, deformable_groups=64):
		super(AdaptBlockFeat, self).__init__()
		self.opt = opt
		self.mask = True

		regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
									   [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
		self.register_buffer('regular_matrix', regular_matrix.float())

		self.concat = conv(inplanes*2, inplanes*2, groups=inplanes*2, mode='CL')
		self.concat2 = conv(inplanes*2, inplanes, groups=inplanes, mode='CL')

		self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
		self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
		self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
			padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
		self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True) 

	def forward(self, x, h_hr, feat):
		N, _, H, W = x.shape
		x_h_hr = self.concat2(self.concat(torch.cat([x, h_hr], dim=1)))

		transform_matrix = self.transform_matrix_conv(x_h_hr)
		transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
		offset = torch.matmul(transform_matrix, self.regular_matrix)
		offset = offset - self.regular_matrix
		offset = offset.transpose(1,2).reshape((N,H,W,18)).permute(0,3,1,2)

		translation = self.translation_conv(x_h_hr)
		offset[:,0::2,:,:] += translation[:,0:1,:,:]
		offset[:,1::2,:,:] += translation[:,1:2,:,:]
		
		out = self.adapt_conv(feat, offset) 
		out = self.relu(out)
		return out

class AdaptBlockOffset(nn.Module):
	def __init__(self, opt, inplanes=64, outplanes=64, stride=1, dilation=1, deformable_groups=64):
		super(AdaptBlockOffset, self).__init__()
		self.D = deformable_groups
		self.opt = opt

		regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
									   [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
		self.register_buffer('regular_matrix', regular_matrix.float())

		self.concat = conv(inplanes*2, inplanes*2, groups=inplanes*2, mode='CL')
		self.concat2 = conv(inplanes*2, inplanes, groups=inplanes, mode='CL')

		self.transform_matrix_conv = nn.Conv2d(inplanes, 4*self.D, 5, 1, 2, bias=True)
		self.translation_conv = nn.Conv2d(inplanes, 2*self.D, 5, 1, 2, bias=True)
		self.mask_conv = nn.Conv2d(inplanes, 9*self.D, 5, 1, 2, bias=True)
		self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True) 

	def forward(self, x, h_hr):
		N, _, H, W = x.shape
		x_h_hr = self.concat2(self.concat(torch.cat([x, h_hr], dim=1)))

		transform_matrix = self.transform_matrix_conv(x_h_hr)
		transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,self.D,2,2))
		offset = torch.matmul(transform_matrix, self.regular_matrix)
		offset = offset - self.regular_matrix
		offset = offset.transpose(2,3).reshape((N,H,W,self.D,18)).permute(0,3,4,1,2) # N D C H W

		translation = self.translation_conv(x_h_hr).reshape(N,self.D,2,H,W)
		offset[:,:,0::2,:,:] += translation[:,:,0:1,:,:]
		offset[:,:,1::2,:,:] += translation[:,:,1:2,:,:]

		mask = self.mask_conv(x_h_hr).contiguous()
		mask = torch.sigmoid(mask)
		
		return offset.reshape(N,self.D*18, H, W).contiguous(), mask


class AdaptBlock2_3x3(nn.Module):
	def __init__(self, opt, inplanes=64, outplanes=64, stride=1, dilation=1, deformable_groups=64):
		super(AdaptBlock2_3x3, self).__init__()
		self.opt = opt
		self.mask = True

		regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
									   [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
		self.register_buffer('regular_matrix', regular_matrix.float())

		self.concat = conv(inplanes*2, inplanes*2, groups=inplanes*2, mode='CL')
		self.concat2 = conv(inplanes*2, inplanes, groups=inplanes, mode='CL')

		self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
		self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)

	def forward(self, x, h_hr):
		N, _, H, W = x.shape
		x_h_hr = self.concat2(self.concat(torch.cat([x, h_hr], dim=1)))

		transform_matrix = self.transform_matrix_conv(x_h_hr)
		transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
		offset = torch.matmul(transform_matrix, self.regular_matrix)
		offset = offset - self.regular_matrix
		offset = offset.transpose(1,2).reshape((N,H,W,18)).permute(0,3,1,2)

		translation = self.translation_conv(x_h_hr)
		offset[:,0::2,:,:] += translation[:,0:1,:,:]
		offset[:,1::2,:,:] += translation[:,1:2,:,:]

		return  offset
		
class ResBlock_Pre(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', predict=False):
		super(ResBlock_Pre, self).__init__()
		assert in_channels == out_channels
		self.predict = predict
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size,
						  stride, padding=padding, bias=bias, mode=mode)

		if self.predict:
			mlp = [conv(64, 4, 1, padding=0, mode='CR'),
				   conv(4, 64, 1, padding=0, mode='C')]
			self.mlp = seq(mlp)
		
	def forward(self, x, p=None):
		x_in = x.clone()
		if self.predict:
			kernel = self.mlp(p)
		res = self.res(x_in)
		return x + res * kernel

class ResBlock(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC'):
		super(ResBlock, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size,
						  stride, padding=padding, bias=bias, mode=mode)
		
	def forward(self, x):
		x_in = x.clone()
		res = self.res(x_in)
		return x + res

class Predictor(nn.Module):
	def __init__(self, opt):
		super(Predictor, self).__init__()
		self.scale = opt.scale
		self.mean = MeanShift()
		
		head = [conv(3+3, 64, 3, 1, mode='CR')]
		self.head = seq(head)

		predictor = [conv(128, 64, 3, 1, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 nn.AdaptiveAvgPool2d(1) ]
		self.predictor = seq(predictor)
	
	def forward(self, lr, hr, concat):
		up_lr = F.interpolate(lr, size=hr.shape[2:], mode='bilinear', align_corners=True)
		up_lr = self.mean(up_lr)
		hr = self.mean(hr)
		lr_hr_center = torch.cat([up_lr, hr], dim=1)
		
		h = self.head(lr_hr_center)
		concat_up = F.interpolate(concat, size=h.shape[2:], mode='bilinear', align_corners=True)
		input = torch.cat([h, concat_up], 1)

		out = self.predictor(input)
		return out

class DownBlock(nn.Module):
	def __init__(self, scale):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		n, c, h, w = x.size()
		x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
		x = x.view(n, c*(self.scale**2), h//self.scale, w//self.scale)
		return x
		
class CALayer(nn.Module):
	def __init__(self, channel=64, reduction=16):
		super(CALayer, self).__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y 

class RCABlock(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', reduction=16):
		super(RCABlock, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size,
						stride, padding, bias=bias, mode=mode)
		self.ca = CALayer(out_channels, reduction)

	def forward(self, x):
		res = self.res(x)
		res = self.ca(res) 
		return res + x

class RCAGroup(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
		super(RCAGroup, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
					   bias, mode, reduction) for _ in range(nb)]
		RG.append(conv(out_channels, out_channels, mode='C'))

		self.rg = nn.Sequential(*RG)

	def forward(self, x):
		res = self.rg(x)
		return res + x


class ResGroup(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
		super(ResGroup, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]
		# RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
		# 			   bias, mode, reduction) for _ in range(nb)]
		RG = [ResBlock(in_channels, out_channels, kernel_size, stride, padding,
					   bias, mode) for _ in range(nb)]

		self.rg = nn.Sequential(*RG)

	def forward(self, x, res=None):
		res = self.rg(x)
		return res + x

class ResGroup_pre(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
		super(ResGroup_pre, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]
		self.nb = nb
		# RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
		# 			   bias, mode, reduction) for _ in range(nb)]
		for i in range(nb):
			setattr(self, 'block%d'%i, ResBlock_Pre(in_channels, out_channels, kernel_size, stride, padding,
					   bias, mode, True))

	def forward(self, res, pre):
		for i in range(self.nb):
			res = getattr(self, 'block%d'%i)(res, pre)
		return res

class ContrasExtractorLayer(nn.Module):
	def __init__(self, n_feat=64):
		super(ContrasExtractorLayer, self).__init__()
		vgg16_layers = [
			'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
			'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
			'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
			'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
			'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
			'pool5'
		]
		conv3_1_idx = vgg16_layers.index('conv3_1')
		features = getattr(vgg, 'vgg16')(pretrained=True).features[:conv3_1_idx + 1]
		modified_net = OrderedDict()
		for k, v in zip(vgg16_layers, features):
			modified_net[k] = v

		modified_net.pop('pool1')
		modified_net.pop('pool2')
		self.tail = nn.Conv2d(256, n_feat, 3, 1, 1)

		self.model = nn.Sequential(modified_net)
		# the mean is for image with range [0, 1]
		self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		# the std is for image with range [0, 1]
		self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

	def forward(self, batch):
		batch = (batch - self.mean) / self.std
		output = self.tail(self.model(batch))
		return output


class Flownet(nn.Module):
	def __init__(self, in_channels):
		super(Flownet, self).__init__()
		self.conv_first = conv(in_channels=in_channels * 2, out_channels=in_channels, mode='CL')
		self.conv_second = conv(in_channels=in_channels, out_channels=in_channels, mode='CL')
		self.trans = conv(in_channels=in_channels, out_channels=2, mode='C')
	def forward(self, x, y):
		out = torch.cat([x, y], dim=1)
		out = self.trans(self.conv_first(out))
		return out

class TransOffsetworelu(nn.Module):
	def __init__(self):
		super(TransOffsetworelu, self).__init__()
		self.conv_first = conv(18, 2, mode='C')
	def forward(self, offset):
		return self.conv_first(offset)

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

class MultiAdSTN(ModulatedDeformConv2d):
	def __init__(self, opt, inplanes=64, outplanes=64, stride=1, dilation=1, deformable_groups=64):
		super(MultiAdSTN, self).__init__(inplanes, 
										 outplanes, 
										 kernel_size=3, 
										 padding=1,
										 stride=stride,
										 dilation=dilation,
										 deform_groups=deformable_groups)
		self.opt = opt
		self.flow_l1 = AdaptBlock2_3x3(opt, inplanes=inplanes, outplanes=outplanes, stride=stride, dilation=dilation, deformable_groups=deformable_groups)
		self.flow_l2 = AdaptBlock2_3x3(opt, inplanes=inplanes, outplanes=outplanes, stride=stride, dilation=dilation, deformable_groups=deformable_groups)
		self.flow_l3 = AdaptBlock2_3x3(opt, inplanes=inplanes, outplanes=outplanes, stride=stride, dilation=dilation, deformable_groups=deformable_groups)
		self.adastn = AdaptBlockOffset(opt, inplanes=inplanes, outplanes=outplanes, stride=stride, dilation=dilation, deformable_groups=deformable_groups)
		self.trans_l3 = TransOffsetworelu()
		self.trans_l2 = TransOffsetworelu()
		self.trans_l1 = TransOffsetworelu()
		self.center = opt.n_frame//2
		self.backwarp_tenGrid = {}
		self.backwarp_tenPartial = {}
	

	def forward(self, nbr_feat_l, ref_feat_l, feat_prop, offset, flag=False):
		coe = 1
		offset = coe * offset
		offset_down4 = F.interpolate(offset, scale_factor=0.25, align_corners=True, mode='bilinear') / 4.
		offset_down2 = F.interpolate(offset, scale_factor=0.5, align_corners=True, mode='bilinear') / 2.
		if not flag:
			for i in range(3, 0, -1):
				if i == 3:
					warp_down4 = flow_warp(nbr_feat_l[i-1], offset_down4)
					offset_p1 = getattr(self, 'flow_l%d'%i)(warp_down4, ref_feat_l[i-1]) # deformable offset
					offset_p1 = getattr(self, 'trans_l%d'%i)(offset_p1) # flow
					offset_p1_up2 = F.interpolate(offset_p1, scale_factor=2, align_corners=True, mode='bilinear') * 2
				elif i == 2:
					warp_down2 = flow_warp(nbr_feat_l[i-1], offset_down2+offset_p1_up2)
					offset_p2 = getattr(self, 'flow_l%d'%i)(warp_down2, ref_feat_l[i-1]) # p2
					offset_p2 = getattr(self, 'trans_l%d'%i)(offset_p2) # flow
					offset_p2_up2 = F.interpolate(offset_p2+offset_p1_up2, scale_factor=2, align_corners=True, mode='bilinear') * 2
				else:
					warp_ = flow_warp(nbr_feat_l[i-1], offset+offset_p2_up2)
					# warp_ = (1-mask_) * ref_feat_l[i-1] + warp_
					offset_p3 = getattr(self, 'flow_l%d'%i)(warp_, ref_feat_l[i-1]) # p3
					offset_p3 = getattr(self, 'trans_l%d'%i)(offset_p3) # flow
					offset = offset_p3 + offset_p2_up2 + offset
		
		nbr = flow_warp(nbr_feat_l[0], offset)

		feat = flow_warp(feat_prop, offset).contiguous()

		de_offset, mask = self.adastn(nbr, ref_feat_l[0])

		out = modulated_deform_conv2d(feat, de_offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
		return out

class OffRes(nn.Module):
	def __init__(self, n_feats=64):
		super(OffRes, self).__init__()
		off_pre = [conv(n_feats * 2, n_feats, mode='CL'),
						conv(n_feats, n_feats, mode='CL'),
						conv(n_feats, n_feats, mode='CL'),
						conv(n_feats, 2, mode='C')]
		self.off_pre = seq(off_pre)
		self.backwarp_tenGrid = {}
		self.backwarp_tenPartial = {}

	def forward(self, offset, first, center):
		first_out, mask = self.get_backwarp(first, offset)
		first_out = first_out + (1 - mask) * center
		off_res = self.off_pre(torch.cat([first_out, center], dim=1))
		return offset + off_res
		
	
	def backwarp(self, tenInput, tenFlow):
		index = str(tenFlow.shape) + str(tenInput.device)
		if index not in self.backwarp_tenGrid:
			tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
									tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
			tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
									tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
			self.backwarp_tenGrid[index] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

		if index not in self.backwarp_tenPartial:
			self.backwarp_tenPartial[index] = tenFlow.new_ones([ tenFlow.shape[0], 
													1, tenFlow.shape[2], tenFlow.shape[3] ])

		tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
							tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
		tenInput = torch.cat([ tenInput, self.backwarp_tenPartial[index] ], 1)

		tenOutput = torch.nn.functional.grid_sample(input=tenInput, 
					grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
					mode='bilinear', padding_mode='zeros', align_corners=True)

		return tenOutput

	def get_backwarp(self, tenFeature, flow):
		tenoutput = self.backwarp(tenFeature, flow) 
		tenMask = tenoutput[:, -1:, :, :]
		tenMask[tenMask > 0.999] = 1.0
		tenMask[tenMask < 1.0] = 0.0
		return tenoutput[:, :-1, :, :] * tenMask, tenMask

class SPYAdaSTN(nn.Module):
	def __init__(self, opt, inplanes=64, outplanes=64, stride=1, dilation=1, deformable_groups=64):
		super(SPYAdaSTN, self).__init__()
		self.opt = opt
		self.adastn = AdaptBlockFeat(opt, inplanes=inplanes, outplanes=outplanes, stride=stride, dilation=dilation, deformable_groups=deformable_groups)


	def forward(self, nbr_feat_l, ref_feat_l, feat_prop, offset):

		nbr = flow_warp(nbr_feat_l[0], offset)

		feat = flow_warp(feat_prop, offset)

		feat = self.adastn(nbr, ref_feat_l[0], feat)

		return feat


def flow_warp(x,
			  flow,
			  interpolation='bilinear',
			  padding_mode='zeros',
			  align_corners=True):
	"""Warp an image or a feature map with optical flow.
	Args:
		x (Tensor): Tensor with size (n, c, h, w).
		flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
			a two-channel, denoting the width and height relative offsets.
			Note that the values are not normalized to [-1, 1].
		interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
			Default: 'bilinear'.
		padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
			Default: 'zeros'.
		align_corners (bool): Whether align corners. Default: True.
	Returns:
		Tensor: Warped image or feature map.
	"""
	flow = flow.permute(0, 2, 3, 1)
	if x.size()[-2:] != flow.size()[1:3]:
		raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
						 f'flow ({flow.size()[1:3]}) are not the same.')
	_, _, h, w = x.size()
	# create mesh grid
	grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
	grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
	grid.requires_grad = False

	grid_flow = grid + flow
	# scale grid_flow to [-1,1]
	grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
	grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
	grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
	output = F.grid_sample(
		x,
		grid_flow,
		mode=interpolation,
		padding_mode=padding_mode,
		align_corners=align_corners)
	return output