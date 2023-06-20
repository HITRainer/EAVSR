import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
from util.util import *
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from . import pwc_net
import time as Time
import os
from models import pwc_net


class EAVSRPx2Model(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(EAVSRPx2Model, self).__init__(opt)
		# torch.autograd.set_detect_anomaly(True)

		self.opt = opt
		self.scale = opt.scale
		self.visual_names = ['data_lr_seq', 'data_hr_seq', 'data_sr_seq']

		self.loss_names = ['EAVSRP_L1', 'EAVSRP_Total'] 
		self.model_names = ['EAVSRP'] 
		self.optimizer_names = ['EAVSRP_optimizer_%s' % opt.optimizer]
		
		eavsrp = EAVSRP(opt, './ckpt/spynet_20210409-c6c1bd09.pth')
		self.netEAVSRP= N.init_net(eavsrp, opt.init_type, opt.init_gain, opt.gpu_ids)


		gf = GuidedFilter(opt.patch_size//4)
		self.netGF = N.init_net(gf, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netGF, requires_grad=False)

		
		pwcnet = pwc_net.PWCNET()
		self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netPWCNET, requires_grad=False)
		
		if self.isTrain:	
			align_id = []
			for module_name in ['backward_1', 'forward_1', 'backward_2', 'forward_2']:
				align_id += list(map(id, self.netEAVSRP.module.deform_align[module_name].parameters()))
		
			basic_para = filter(lambda p: id(p) not in align_id,
					 			self.netEAVSRP.module.parameters())
			align_para = filter(lambda p: id(p) in align_id,
					 			self.netEAVSRP.module.parameters())

			self.optimizer_EAVSRP = optim.Adam([{'params': basic_para}, #change 16-
										  	   {'params': align_para, 'lr': 1e-5}],
										  		lr=opt.lr,
										  		betas=(opt.beta1, opt.beta2),
										  		weight_decay=opt.weight_decay)		
			

			self.optimizers = [self.optimizer_EAVSRP]

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)

		else:
			self.time = 0.0
			self.isfirst = True
			self.num = 0

	def set_input(self, input, epoch):
		self.data_lr_seq = input['lr_seq'].to(self.device)
		self.data_hr_seq = input['hr_seq'].to(self.device)
		self.image_name = input['fname']
		#visualize
		self.idx = random.randint(0, self.opt.n_frame-1)
		self.data_lr = self.data_lr_seq[:, self.idx,...]
		#post-processing
		self.epoch = epoch
		

	def forward(self):
		if self.isTrain:
			self.data_sr_seq = self.netEAVSRP(self.data_lr_seq)
			self.data_sr = self.data_sr_seq[:, self.idx,...]
			if self.epoch >= self.opt.npost:
				self.mask = []
				self.data_hr_align = []
				for idx in range(self.data_hr_seq.shape[1]):
					gf_out = self.netGF(self.data_hr_seq[:, idx, ...], F.interpolate(self.data_lr_seq[:, idx, ...], \
																					 scale_factor=self.opt.scale, mode='bilinear'))
					align_out, mask = self.get_backwarp(self.data_lr_seq[:, idx, ...], \
																			self.data_hr_seq[:, idx, ...], self.netPWCNET, scale=self.opt.scale)
					self.data_hr_align.append(align_out)
					self.mask.append(mask)
				self.data_hr_align = torch.stack(self.data_hr_align, dim=1)
				self.mask = torch.stack(self.mask, dim=1)
				self.data_sr_seq = self.data_sr_seq * self.mask
			self.data_hr = self.data_hr_seq[:, self.idx,...]
		else:
			start = time.time()
			self.data_sr_seq = self.netEAVSRP(self.data_lr_seq)
			self.data_sr = self.data_sr_seq[:, self.idx,...]
			end = time.time()
			if not self.isfirst:
				self.time += end - start
				self.num += 1
			self.isfirst = False
	
	def backward(self):  
		self.loss_EAVSRP_L1 = self.criterionL1(self.data_hr_seq, self.data_sr_seq).mean()
		self.loss_EAVSRP_Total = self.loss_EAVSRP_L1

		self.loss_EAVSRP_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_EAVSRP.zero_grad()
		self.backward()
		self.optimizer_EAVSRP.step() 
	
class EAVSRP(nn.Module):
	def __init__(self, opt, spynet_pretrained=None):
		super(EAVSRP, self).__init__()
		self.opt = opt
		self.predict = opt.predict
		self.n_resblock = 30
		self.n_frame = opt.n_frame
		self.n_feats = 64
		self.n_flow = opt.n_flow
		
		self.spynet = SPyNet(pretrained=spynet_pretrained)
		for param in self.spynet.parameters():
					param.requires_grad = False
		#feature extractor
		self.encoder = N.ContrasExtractorLayer(self.n_feats)

		#propagation and alignment module
		self.deform_align = nn.ModuleDict()
		self.backbone = nn.ModuleDict()
		self.fusion = nn.ModuleDict()
		modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
		for i, module in enumerate(modules):
			self.deform_align[module] = N.MultiAdSTN(opt, self.n_feats, self.n_feats, deformable_groups=8)
			self.backbone[module] = ResidualBlocksWithInputConv(
				(2 + i) * self.n_feats, self.n_feats, self.n_resblock)
			self.fusion[module] = nn.Conv2d(self.n_feats*3, self.n_feats, 1, 1, 0, bias=True)

		# upsample
		self.reconstruction = ResidualBlocksWithInputConv(
			5 * self.n_feats, self.n_feats, 5)
		self.upsample1 = N.seq([N.conv(self.n_feats, self.n_feats*2*2, mode='C'),
						 nn.PixelShuffle(2)])
		self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
		self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
		self.img_upsample = nn.Upsample(
			scale_factor=2, mode='bilinear', align_corners=False)

		# activation function
		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

		

	def check_if_mirror_extended(self, lrs):
		"""Check whether the input is a mirror-extended sequence.
		If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
		(t-1-i)-th frame.
		Args:
			lrs (tensor): Input LR images with shape (n, t, c, h, w)
		"""

		self.is_mirror_extended = False
		if lrs.size(1) % 2 == 0:
			lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
			if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
				self.is_mirror_extended = True	
	
	def compute_flow(self, lrs):
		"""Compute optical flow using SPyNet for feature warping.
		Note that if the input is an mirror-extended sequence, 'flows_forward'
		is not needed, since it is equal to 'flows_backward.flip(1)'.
		Args:
			lrs (tensor): Input LR images with shape (n, t, c, h, w)
		Return:
			tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
				flows used for forward-time propagation (current to previous).
				'flows_backward' corresponds to the flows used for
				backward-time propagation (current to next).
		"""

		n, t, c, h, w = lrs.shape
		lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
		lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

		flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w) # lrs_2 -> lrs_1

		flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w) # lrs_1 -> lrs_2

		return flows_forward, flows_backward
		
	def forward(self, lrs):
		n, t, c, h, w = lrs.shape
		assert h >= 64 and w >= 64, (
			'The height and width of inputs should be at least 64, '
			f'but got {h} and {w}.')

		# compute optical flow
		with torch.no_grad():
			flows_forward, flows_backward = self.compute_flow(lrs)
			# end = Time.time()
		feats = {}

		lr_flatten = lrs.view(-1, c, h, w)
		#L1
		lr_feature = self.encoder(lr_flatten)
		#L2
		lr_feature_down2 = F.interpolate(lr_feature, scale_factor=0.5, mode='bilinear', align_corners=False)
		#L3
		lr_feature_down4 = F.interpolate(lr_feature, scale_factor=0.25, mode='bilinear', align_corners=False)
		
		lr_feature = lr_feature.view(n, t, -1, h, w)
		lr_feature_down2 = lr_feature_down2.view(n, t, -1, h//2, w//2)
		lr_feature_down4 = lr_feature_down4.view(n, t, -1, h//4, w//4)
		feats['spatial'] = [lr_feature[:, i, ...] for i in range(0, t)]
		feats['spatial_d2'] = [lr_feature_down2[:, i, ...] for i in range(0, t)]
		feats['spatial_d4'] = [lr_feature_down4[:, i, ...] for i in range(0, t)]

		for iter_ in [1, 2]:
			for direction in ['backward', 'forward']:
				module = f'{direction}_{iter_}'
				feats[module] = []

				if direction == 'backward':
					flows = flows_backward
				else:
					flows = flows_forward
				feats = self.propagate(feats, flows, module)
		out = self.upsample(lrs, feats)
		return out
	
	def propagate(self, feats, flows, module_name):
		"""Propagate the latent features throughout the sequence.

		Args:
			feats dict(list[tensor]): Features from previous branches. Each
				component is a list of tensors with shape (n, c, h, w).
			flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
			module_name (str): The name of the propgation branches. Can either
				be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

		Return:
			dict(list[tensor]): A dictionary containing all the propagated
				features. Each key in the dictionary corresponds to a
				propagation branch, which is represented by a list of tensors.
		"""
		n, t, _, h, w = flows.size()
		# backward-time propgation

		frame_idx = range(0, t + 1)
		flow_idx = range(-1, t)
		mapping_idx = list(range(0, len(feats['spatial']))) # [0, 7)
		mapping_idx += mapping_idx[::-1] # [0, ... 6, 6, ... 0]

		if 'backward' in module_name:
			frame_idx = frame_idx[::-1] # [6,... 0]
			flow_idx = frame_idx # [6,... 0]
		feat_prop = flows.new_zeros(n, self.n_feats, h, w)
		
		# Start to propagete
		for i, idx in enumerate(frame_idx):
			feat_current = feats['spatial'][mapping_idx[idx]]
			feat_current_down2 = feats['spatial_d2'][mapping_idx[idx]]
			feat_current_down4 = feats['spatial_d4'][mapping_idx[idx]]

			# second-order deformable alignment
			if i > 0:
				current_feat = [feat_current, feat_current_down2, feat_current_down4]
				if 'backward' in module_name:
					nbr_feat = [feats['spatial'][mapping_idx[idx + 1]],
								feats['spatial_d2'][mapping_idx[idx + 1]],
								feats['spatial_d4'][mapping_idx[idx + 1]]]
				else:
					nbr_feat = [feats['spatial'][mapping_idx[idx - 1]],
								feats['spatial_d2'][mapping_idx[idx - 1]],
								feats['spatial_d4'][mapping_idx[idx - 1]]]
				flow_n1 = flows[:, flow_idx[i], :, :, :]

				cond_n1 = self.deform_align[module_name](nbr_feat, current_feat, feat_prop, flow_n1)

				# initialize second-order features
				feat_n2 = torch.zeros_like(feat_prop)
				flow_n2 = torch.zeros_like(flow_n1)
				cond_n2 = torch.zeros_like(cond_n1)

				if i > 1:  # second-order features
					feat_n2 = feats[module_name][-2]
					if 'backward' in module_name:
						nbr_feat = [feats['spatial'][mapping_idx[idx + 2]],
									feats['spatial_d2'][mapping_idx[idx + 2]],
									feats['spatial_d4'][mapping_idx[idx + 2]]]
					else:
						nbr_feat = [feats['spatial'][mapping_idx[idx - 2]],
									feats['spatial_d2'][mapping_idx[idx - 2]],
									feats['spatial_d4'][mapping_idx[idx - 2]]]

					flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

					flow_n2 = flow_n1 + flow_warp(flow_n2,
												  flow_n1.permute(0, 2, 3, 1))
					cond_n2 = self.deform_align[module_name](nbr_feat, current_feat, feat_n2, flow_n2)

				feat_prop = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
				feat_prop = self.fusion[module_name](feat_prop)
			
			# concatenate and residual blocks
			feat = [feat_current] + [
				feats[k][idx]
				for k in feats if k not in ['spatial', 'spatial_d2', 'spatial_d4', module_name]
			] + [feat_prop]
		
			feat = torch.cat(feat, dim=1)
			feat_prop = feat_prop + self.backbone[module_name](feat)
			feats[module_name].append(feat_prop)

		if 'backward' in module_name:
			feats[module_name] = feats[module_name][::-1]
		
		return feats

	def upsample(self, lqs, feats):
		"""Compute the output image given the features.

		Args:
			lqs (tensor): Input low quality (LQ) sequence with
				shape (n, t, c, h, w).
			feats (dict): The features from the propgation branches.

		Returns:
			Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

		"""

		outputs = []
		num_outputs = len(feats['spatial'])

		mapping_idx = list(range(0, num_outputs))
		mapping_idx += mapping_idx[::-1]

		for i in range(0, lqs.size(1)):
			hr = [feats[k].pop(0) for k in feats if k != 'spatial' and k != 'spatial_d2' and k != 'spatial_d4']
			hr.insert(0, feats['spatial'][mapping_idx[i]])
			hr = torch.cat(hr, dim=1)

			hr = self.reconstruction(hr)
			hr = self.lrelu(self.upsample1(hr))
			hr = self.lrelu(self.conv_hr(hr))
			hr = self.conv_last(hr)
			hr += self.img_upsample(lqs[:, i, :, :, :])

			outputs.append(hr)

		return torch.stack(outputs, dim=1)


class ResidualBlocksWithInputConv(nn.Module):
	"""Residual blocks with a convolution in front.
	Args:
		in_channels (int): Number of input channels of the first conv.
		out_channels (int): Number of channels of the residual blocks.
			Default: 64.
		num_blocks (int): Number of residual blocks. Default: 30.
	"""

	def __init__(self, in_channels, out_channels=64, num_blocks=30):
		super().__init__()

		main = []

		# a convolution used to match the channels of the residual blocks
		main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
		main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

		# residual blocks
		main.append(N.RCAGroup(out_channels, out_channels, nb=num_blocks))
		# main.append(
		#     make_layer(
		#         ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

		self.main = nn.Sequential(*main)

	def forward(self, feat):
		"""
		Forward function for ResidualBlocksWithInputConv.
		Args:
			feat (Tensor): Input feature with shape (n, in_channels, h, w)
		Returns:
			Tensor: Output feature with shape (n, out_channels, h, w)
		"""
		return self.main(feat)

class SPyNet(nn.Module):
	"""SPyNet network structure.
	The difference to the SPyNet in [tof.py] is that
		1. more SPyNetBasicModule is used in this version, and
		2. no batch normalization is used in this version.
	Paper:
		Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
	Args:
		pretrained (str): path for pre-trained SPyNet. Default: None.
	"""

	def __init__(self, pretrained):
		super().__init__()

		self.basic_module = nn.ModuleList(
			[SPyNetBasicModule() for _ in range(6)])

		if isinstance(pretrained, str):
			logger = get_root_logger()
			load_checkpoint(self, pretrained, strict=True, logger=logger)
		elif pretrained is not None:
			raise TypeError('[pretrained] should be str or None, '
							f'but got {type(pretrained)}.')

		self.register_buffer(
			'mean',
			torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer(
			'std',
			torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

	def compute_flow(self, ref, supp):
		"""Compute flow from ref to supp.
		Note that in this function, the images are already resized to a
		multiple of 32.
		Args:
			ref (Tensor): Reference image with shape of (n, 3, h, w).
			supp (Tensor): Supporting image with shape of (n, 3, h, w).
		Returns:
			Tensor: Estimated optical flow: (n, 2, h, w).
		"""
		n, _, h, w = ref.size()

		# normalize the input images
		ref = [(ref - self.mean) / self.std]
		supp = [(supp - self.mean) / self.std]

		# generate downsampled frames
		for level in range(5):
			ref.append(
				F.avg_pool2d(
					input=ref[-1],
					kernel_size=2,
					stride=2,
					count_include_pad=False))
			supp.append(
				F.avg_pool2d(
					input=supp[-1],
					kernel_size=2,
					stride=2,
					count_include_pad=False))
		ref = ref[::-1]
		supp = supp[::-1]

		# flow computation
		flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
		for level in range(len(ref)):
			if level == 0:
				flow_up = flow
			else:
				flow_up = F.interpolate(
					input=flow,
					scale_factor=2,
					mode='bilinear',
					align_corners=True) * 2.0

			# add the residue to the upsampled flow
			flow = flow_up + self.basic_module[level](
				torch.cat([
					ref[level],
					flow_warp(
						supp[level],
						flow_up.permute(0, 2, 3, 1),
						padding_mode='border'), flow_up
				], 1))

		return flow

	def forward(self, ref, supp):
		"""Forward function of SPyNet.
		This function computes the optical flow from ref to supp.
		Args:
			ref (Tensor): Reference image with shape of (n, 3, h, w).
			supp (Tensor): Supporting image with shape of (n, 3, h, w).
		Returns:
			Tensor: Estimated optical flow: (n, 2, h, w).
		"""

		# upsize to a multiple of 32
		h, w = ref.shape[2:4]
		w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
		h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
		ref = F.interpolate(
			input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
		supp = F.interpolate(
			input=supp,
			size=(h_up, w_up),
			mode='bilinear',
			align_corners=False)

		# compute flow, and resize back to the original resolution
		flow = F.interpolate(
			input=self.compute_flow(ref, supp),
			size=(h, w),
			mode='bilinear',
			align_corners=False)

		# adjust the flow values
		flow[:, 0, :, :] *= float(w) / float(w_up)
		flow[:, 1, :, :] *= float(h) / float(h_up)

		return flow	

class SPyNetBasicModule(nn.Module):
	"""Basic Module for SPyNet.
	Paper:
		Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
	"""

	def __init__(self):
		super().__init__()

		self.basic_module = nn.Sequential(
			ConvModule(
				in_channels=8,
				out_channels=32,
				kernel_size=7,
				stride=1,
				padding=3,
				norm_cfg=None,
				act_cfg=dict(type='ReLU')),
			ConvModule(
				in_channels=32,
				out_channels=64,
				kernel_size=7,
				stride=1,
				padding=3,
				norm_cfg=None,
				act_cfg=dict(type='ReLU')),
			ConvModule(
				in_channels=64,
				out_channels=32,
				kernel_size=7,
				stride=1,
				padding=3,
				norm_cfg=None,
				act_cfg=dict(type='ReLU')),
			ConvModule(
				in_channels=32,
				out_channels=16,
				kernel_size=7,
				stride=1,
				padding=3,
				norm_cfg=None,
				act_cfg=dict(type='ReLU')),
			ConvModule(
				in_channels=16,
				out_channels=2,
				kernel_size=7,
				stride=1,
				padding=3,
				norm_cfg=None,
				act_cfg=None))

	def forward(self, tensor_input):
		"""
		Args:
			tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
				8 channels contain:
				[reference image (3), neighbor image (3), initial flow (2)].
		Returns:
			Tensor: Refined flow with shape (b, 2, h, w)
		"""
		return self.basic_module(tensor_input)

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


class FLOW(nn.Module):
	def __init__(self, opt, net):
		super(FLOW, self).__init__()
		self.opt = opt
		self.netFlow = net
		self.n_frame = opt.n_frame
		self.center_frame_idx = self.n_frame//2 # index of key frame
		self.n_flow = opt.n_flow
	
	def forward(self, lr_seq):
		off_f = []
		times = lr_seq.shape[1] // self.n_flow
		for time in range(times):
			curr = lr_seq[:, time*self.n_flow+1, ...] # align7
			last = lr_seq[:, time*self.n_flow, ...]
			off_f.append(self.netFlow(curr, last))
		return off_f

def diff_y(input, r):
	assert input.dim() == 4

	left = input[:, :, :, r:2 * r + 1]
	middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
	right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1: -r - 1]

	output = torch.cat([left, middle, right], dim=3)

	return output

def diff_x(input, r):
	assert input.dim() == 4

	left = input[:, :, r:2 * r + 1]
	middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
	right = input[:, :, -1:] - input[:, :, -2 * r - 1: -r - 1]

	output = torch.cat([left, middle, right], dim=2)

	return output

class BoxFilter(nn.Module):
	def __init__(self, r):
		super(BoxFilter, self).__init__()
		self.r = r

	def forward(self, x):
		assert x.dim() == 4
		return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

class GuidedFilter(nn.Module):
	def __init__(self, r, eps=1e-8):
		super(GuidedFilter, self).__init__()
		self.r = r
		self.eps = eps
		self.boxfilter = BoxFilter(r)

	def forward(self, x, y):
		n_x, c_x, h_x, w_x = x.size()
		n_y, c_y, h_y, w_y = y.size()

		assert n_x == n_y
		assert c_x == 1 or c_x == c_y
		assert h_x == h_y and w_x == w_y
		assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

		# N
		N = self.boxfilter(torch.tensor(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
		mean_x = self.boxfilter(x) / N
		mean_y = self.boxfilter(y) / N
		# cov_xy
		cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
		# var_x
		var_x = self.boxfilter(x * x) / N - mean_x * mean_x

		# A
		A = cov_xy / (var_x + self.eps)
		# b
		b = mean_y - A * mean_x
		mean_A = self.boxfilter(A) / N
		mean_b = self.boxfilter(b) / N

		return mean_A * x + mean_b
