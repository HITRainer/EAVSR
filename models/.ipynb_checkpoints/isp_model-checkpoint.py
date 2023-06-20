import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L
from .ispjoint_model import ISPNet
from util.util import *
from . import pwc_net

class ISPModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(ISPModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = [opt.loss, 'SSIM', 'VGG', 'Total']
		self.visual_names = ['dslr_warp', 'data_out', 'dslr_mask']
		self.model_names = ['ISP'] # will rename in subclasses
		self.optimizer_names = ['ISP_optimizer_%s' % opt.optimizer]

		isp = ISPNet(opt)
		self.netISP = N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)

		pwcnet = pwc_net.PWCNET()
		self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netPWCNET, requires_grad=False)
        
		if self.isTrain:
			if opt.optimizer == 'Adam':
				self.optimizer = optim.Adam(self.netISP.parameters(),
											lr=opt.lr,
											betas=(opt.beta1, opt.beta2),
											weight_decay=opt.weight_decay)
			elif opt.optimizer == 'SGD':
				self.optimizer = optim.SGD(self.netISP.parameters(),
										   lr=opt.lr,
										   momentum=opt.momentum,
										   weight_decay=opt.weight_decay)
			elif opt.optimizer == 'RMSprop':
				self.optimizer = optim.RMSprop(self.netISP.parameters(),
											   lr=opt.lr,
											   alpha=opt.alpha,
											   momentum=opt.momentum,
											   weight_decay=opt.weight_decay)
			else:
				raise NotImplementedError(
					'optimizer named [%s] is not supported' % opt.optimizer)


			self.optimizers = [self.optimizer]

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)
			self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
		self.isp_coord = {}
        
	def set_input(self, input):
		self.data_raw = input['raw'].to(self.device)
		self.data_dslr = input['dslr'].to(self.device)
		# self.data_dslrmask = input['dslrmask'].to(self.device)
		# self.data_coord = input['index'].to(self.device)
		self.image_paths = input['fname']

	def forward(self):
		N, C, H, W = self.data_raw.shape
		index = str(self.data_raw.shape) + '_' + str(self.data_raw.device)
		if index not in self.isp_coord:
			isp_coord = get_coord(H=H, W=W)
			isp_coord = np.expand_dims(isp_coord, axis=0)
			isp_coord = np.tile(isp_coord, (N, 1, 1, 1))
			# print(isp_coord.shape)
			self.isp_coord[index] = torch.from_numpy(isp_coord).to(self.data_raw.device)
		
		self.data_out = self.netISP(self.data_raw, self.isp_coord[index])

		self.dslr_warp, self.dslr_mask = \
			self.get_backwarp(self.data_out, self.data_dslr, self.netPWCNET)
		if self.isTrain:
			self.data_out = self.data_out * self.dslr_mask

	def backward(self):
		self.loss_L1 = self.criterionL1(self.data_out, self.dslr_warp).mean()
		# self.loss_COORD = self.coord.mean()
		self.loss_SSIM = 1 - self.criterionSSIM(self.data_out, self.dslr_warp).mean()
		self.loss_VGG = self.criterionVGG(self.data_out, self.dslr_warp).mean()
		self.loss_Total = self.loss_L1 + self.loss_VGG + self.loss_SSIM * 0.15

		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer.zero_grad()
		self.backward()
		self.optimizer.step()

class ISP(nn.Module):
	def __init__(self, opt):
		super(ISP, self).__init__()
		self.opt = opt

		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4

		self.pre_coord = PreCoord()

		# self.pre_coord_head = N.seq(
		# 	N.conv(2, ch_1, 1, stride=1, padding=0, mode='C'),
		# )

		self.head = N.seq(
			N.DWTForward(),
			N.conv(4*4, ch_1, mode='C')
		)

		self.down1 = N.seq(
			N.RCAGroup(in_channels=ch_1+2, out_channels=ch_1+2, nb=n_blocks),
			N.conv(ch_1+2, ch_1, mode='C'),
		)

		self.down2 = N.seq(
			N.DWTForward(),
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks)
		)

		self.down3 = N.seq(
			N.DWTForward(),
			N.conv(ch_2*4, ch_3, mode='C')
		)

		self.middle = N.seq(
			N.conv(ch_3, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks)
		)

		self.up1 = N.seq(
			N.conv(ch_3, ch_2*4, mode='C'),
			N.DWTInverse()
		)

		self.up2 = N.seq(
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C'),
			N.DWTInverse()
		)

		self.up3 = N.seq(
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1, mode='C')
		)

		self.tail = N.seq(
			N.DWTInverse(),
			N.conv(ch_1//4, ch_1, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1//4, 3, mode='C')
		)

	def forward(self, x, coord):
		pre_coord = self.pre_coord(x) * 0.1
		
		# # pre_coord[:,0] = torch.clamp(pre_coord[:,0], -1+448/3968, 1-448/3968)
		# # pre_coord[:,1] = torch.clamp(pre_coord[:,1], -1+448/2976, 1-448/2976)
		pre_coord = torch.clamp(pre_coord, -1, 1)

		pre_coord = pre_coord.unsqueeze(dim=2).unsqueeze(dim=3)

		# # print(pre_coord)
		pre_coord = pre_coord + coord
		# # pre_coord = torch.clamp(pre_coord, -1, 1)
		# # pre_coord = coord
		# # pre_coord = self.pre_coord_head(pre_coord)
		# # print(pre_coord)
		
		# x = torch.cat((x, pre_coord), 1)
		h = self.head(x)

		id_h = h
		h = torch.cat((h, pre_coord), 1)

		d1 = self.down1(h)
		d2 = self.down2(d1)
		d3 = self.down3(d2)

		# d3 = torch.cat((d3, pre_coord), 1)
		m = self.middle(d3)
		u1 = self.up1(m) + d2
		u2 = self.up2(u1) + d1
		u3 = self.up3(u2) + id_h
		out = self.tail(u3)
		return out, pre_coord

class PreCoord(nn.Module):
	def __init__(self):
		super(PreCoord, self).__init__()

		self.ch_1 = 64

		self.down = N.seq(
			N.conv(4, self.ch_1, 3, stride=2, padding=0, mode='CR'),
			#nn.SELU(inplace=True),
			#N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRCR'),

			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=2, padding=0, mode='CR'),
			#nn.SELU(inplace=True),
			# N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRCR'),

			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=2, padding=0, mode='CR'),
			#nn.SELU(inplace=True),
			# N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRCR'),

			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=2, padding=0, mode='CR'),
			#nn.SELU(inplace=True)
		)

		self.fc = N.seq(
			nn.Linear(self.ch_1*13*13, 256),
			# nn.PReLU(),
			nn.ReLU(inplace=True),
			#nn.SELU(inplace=True),
			nn.Linear(256, 2)
			# nn.ReLU(inplace=True),
			# nn.Linear(64, 2)
			# nn.Tanh()
		)
		
		# self.load_state_dict(torch.load('./ckpt/ispwarp_coord_base_try/coord.pth')['state_dict'])

	def forward(self, x):
		down = self.down(x)
		# print(down.size())
		down = down.view(-1, self.ch_1*13*13)
		out = self.fc(down)
		return out

class NEWISP(nn.Module):
	def __init__(self, opt):
		super(NEWISP, self).__init__()
		self.opt = opt

		ch_1 = 64
		ch_2 = 128
		ch_3 = 128
		n_blocks = 4

		self.head = N.seq(
			N.conv(4, ch_1, mode='CR')
		)  # h (N, ch_1, H/2, W/2)

		self.down1 = N.seq(
			# N.conv(ch_1, ch_1, mode='CR'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward()
		)  # d1 (N, ch_1*4, H/4, W/4)

		self.down2 = N.seq(
			N.conv(ch_1*4, ch_1, mode='C'),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.DWTForward()
		)  # d2 (N, ch_1*4, H/8, W/8)

		self.down3 = N.seq(
			N.conv(ch_1*4, ch_2, mode='C'),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.DWTForward()
		)  # d3 (N, ch_2*4, H/16, W/16)

		self.middle = N.seq(
			N.conv(ch_2*4, ch_3, mode='C'),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.RCAGroup(in_channels=ch_3, out_channels=ch_3, nb=n_blocks),
			N.conv(ch_3, ch_2*4, mode='C')
		)  # m (N, ch_2*4, H/16, W/16)

		self.up3 = N.seq(
			N.DWTInverse(),
			N.RCAGroup(in_channels=ch_2, out_channels=ch_2, nb=n_blocks),
			N.conv(ch_2, ch_1*4, mode='C')
		)  # u3 (N, ch_1*4, H/8, W/8)

		self.up2 = N.seq(
			N.DWTInverse(),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			N.conv(ch_1, ch_1*4, mode='C')
		)  # u2 (N, ch_1*4, H/4, W/4)

		self.up1 = N.seq(
			N.DWTInverse(),
			N.RCAGroup(in_channels=ch_1, out_channels=ch_1, nb=n_blocks),
			# N.conv(ch_1, ch_1, mode='C')
		)  # u1 (N, ch_1, H/2, W/2)

		self.tail = N.seq(
			N.conv(ch_1, ch_1*4, mode='C'),
			nn.PixelShuffle(upscale_factor=2),
			N.conv(ch_1, 3, mode='C')
		)  # u1 (N, 3, H, W)

	def forward(self, x, dslrmask=None):
		h = self.head(x)
		d1 = self.down1(h)
		d2 = self.down2(d1)
		d3 = self.down3(d2)
		m = self.middle(d3) + d3
		u3 = self.up3(m) + d2
		u2 = self.up2(u3) + d1
		u1 = self.up1(u2) + h
		out = self.tail(u1)

		out_mask = out * dslrmask
		return out_mask
