# -*- coding: utf-8 -*-
import torchvision.models.vgg as vgg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp, sqrt
from torch.nn import L1Loss, MSELoss
from torchvision import models
from util.util import grid_positions, warp

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(
			-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) \
		for x in range(window_size)])
	return gauss / gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(
		channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(
		img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(
		img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(
		img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
			   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)

	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)

	return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIMLoss(nn.Module):
	def __init__(self, window_size=11, size_average=True):
		super(SSIMLoss, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and \
				self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size,
					 channel, self.size_average)

def normalize_batch(batch):
	mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
	std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
	return (batch - mean) / std
	
class VGG19(torch.nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		features = models.vgg19(pretrained=True).features
		self.relu1_1 = torch.nn.Sequential()
		self.relu1_2 = torch.nn.Sequential()

		self.relu2_1 = torch.nn.Sequential()
		self.relu2_2 = torch.nn.Sequential()

		self.relu3_1 = torch.nn.Sequential()
		self.relu3_2 = torch.nn.Sequential()
		self.relu3_3 = torch.nn.Sequential()
		self.relu3_4 = torch.nn.Sequential()

		self.relu4_1 = torch.nn.Sequential()
		self.relu4_2 = torch.nn.Sequential()
		self.relu4_3 = torch.nn.Sequential()
		self.relu4_4 = torch.nn.Sequential()

		self.relu5_1 = torch.nn.Sequential()
		self.relu5_2 = torch.nn.Sequential()
		self.relu5_3 = torch.nn.Sequential()
		self.relu5_4 = torch.nn.Sequential()

		for x in range(2):
			self.relu1_1.add_module(str(x), features[x])

		for x in range(2, 4):
			self.relu1_2.add_module(str(x), features[x])

		for x in range(4, 7):
			self.relu2_1.add_module(str(x), features[x])

		for x in range(7, 9):
			self.relu2_2.add_module(str(x), features[x])

		for x in range(9, 12):
			self.relu3_1.add_module(str(x), features[x])

		for x in range(12, 14):
			self.relu3_2.add_module(str(x), features[x])

		for x in range(14, 16):
			self.relu3_3.add_module(str(x), features[x])

		for x in range(16, 18):
			self.relu3_4.add_module(str(x), features[x])

		for x in range(18, 21):
			self.relu4_1.add_module(str(x), features[x])

		for x in range(21, 23):
			self.relu4_2.add_module(str(x), features[x])

		for x in range(23, 25):
			self.relu4_3.add_module(str(x), features[x])

		for x in range(25, 27):
			self.relu4_4.add_module(str(x), features[x])

		for x in range(27, 30):
			self.relu5_1.add_module(str(x), features[x])

		for x in range(30, 32):
			self.relu5_2.add_module(str(x), features[x])

		for x in range(32, 34):
			self.relu5_3.add_module(str(x), features[x])

		for x in range(34, 36):
			self.relu5_4.add_module(str(x), features[x])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		relu1_1 = self.relu1_1(x)
		relu1_2 = self.relu1_2(relu1_1)

		relu2_1 = self.relu2_1(relu1_2)
		relu2_2 = self.relu2_2(relu2_1)

		relu3_1 = self.relu3_1(relu2_2)
		relu3_2 = self.relu3_2(relu3_1)
		relu3_3 = self.relu3_3(relu3_2)
		relu3_4 = self.relu3_4(relu3_3)

		relu4_1 = self.relu4_1(relu3_4)
		relu4_2 = self.relu4_2(relu4_1)
		relu4_3 = self.relu4_3(relu4_2)
		relu4_4 = self.relu4_4(relu4_3)

		relu5_1 = self.relu5_1(relu4_4)
		relu5_2 = self.relu5_2(relu5_1)
		relu5_3 = self.relu5_3(relu5_2)
		relu5_4 = self.relu5_4(relu5_3)

		out = {
			'relu1_1': relu1_1,
			'relu1_2': relu1_2,

			'relu2_1': relu2_1,
			'relu2_2': relu2_2,

			'relu3_1': relu3_1,
			'relu3_2': relu3_2,
			'relu3_3': relu3_3,
			'relu3_4': relu3_4,

			'relu4_1': relu4_1,
			'relu4_2': relu4_2,
			'relu4_3': relu4_3,
			'relu4_4': relu4_4,

			'relu5_1': relu5_1,
			'relu5_2': relu5_2,
			'relu5_3': relu5_3,
			'relu5_4': relu5_4,
		}
		return out

class VGGLoss(nn.Module):
	def __init__(self):
		super(VGGLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = torch.nn.L1Loss()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		content_loss = 0.0
		# # content_loss += self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) * 0.1
		# # content_loss += self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) * 0.2
		content_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2']) * 1
		content_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2']) * 1
		content_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2']) * 2

		return content_loss / 4.

class SPYNetLoss(nn.Module):
	def __init__(self):
		super(SPYNetLoss, self).__init__()
	
	def forward(self, flow1, flow2): # flow1 and flow2: n t 2 h w
		n, t, c, h, w = flow1.shape
		flow1 = flow1.reshape(-1, c, h, w)
		flow2 = flow2.reshape(-1, c, h, w)

		delta_x = flow1[:, 0, ...] - flow2[:, 0, ...]
		delta_y = flow1[:, 1, ...] - flow2[:, 1, ...]

		return torch.sqrt(delta_x**2+delta_y**2).mean()


class FlowLoss(nn.Module):
	def __init__(self):
		super(FlowLoss, self).__init__()
		self.l1loss = torch.nn.L1Loss()
	
	def forward(self, flow1, flow2):
		n, t, c, h, w = flow1.shape
		flow1 = flow1.reshape(-1, c, h, w)
		flow2 = flow2.reshape(-1, c, h, w)

		len_flow1 = torch.sqrt(flow1[:, 0, ...]**2+flow1[:, 1, ...]**2)
		len_flow2 = torch.sqrt(flow2[:, 0, ...]**2+flow2[:, 1, ...]**2)
		lenth = self.l1loss(len_flow1, len_flow2)

		angle = 1 - ((flow1[:, 0, ...] * flow2[:, 0, ...] + flow1[:, 1, ...] * flow2[:, 1, ...]) / \
			len_flow1 * len_flow2).mean()
		return lenth + angle







class TextureLoss(nn.Module):
	def __init__(self):
		super(TextureLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = torch.nn.L1Loss()

	def compute_gram(self, x):
		b, ch, h, w = x.size()
		f = x.view(b, ch, w * h)
		f_T = f.transpose(1, 2)
		G = f.bmm(f_T) / (h * w)
		return G

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		style_loss = 0.0
		style_loss += self.criterion(self.compute_gram(x_vgg['relu1_2']), self.compute_gram(y_vgg['relu1_2'])) * 0.2
		style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2'])) * 1
		style_loss += self.criterion(self.compute_gram(x_vgg['relu3_2']), self.compute_gram(y_vgg['relu3_2'])) * 1
		style_loss += self.criterion(self.compute_gram(x_vgg['relu4_2']), self.compute_gram(y_vgg['relu4_2'])) * 2
		style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2'])) * 5

		return style_loss * 0.007

class SWDLoss(nn.Module):
	def __init__(self):
		super(SWDLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = SWD()
		# self.SWD = SWDLocal()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		N, C, H, W = x.shape  # 192*192
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		swd_loss = 0.0
		swd_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'], k=H//4//p) * 1  # H//4=48
		swd_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'], k=H//8//p) * 1  # H//4=24
		swd_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'], k=H//16//p) * 2  # H//4=12

		return swd_loss * 5 / 100.0 # 0.08

class MarginLoss(nn.Module):
	def __init__(self, opt, kl=False):
		super(MarginLoss, self).__init__()
		self.margin = 1.0  # opt['margin']
		self.safe_radius = 4  # opt['safe_radius'] tea_128:8; tea_vgg:4
		self.scaling_steps = 2  # opt['scaling_steps']
		self.temperature = 0.15 # 0.15 # 0.15 -> 1  0.15
		self.distill_weight = 15 # 15. ->  0.50  0.20
		self.perturb = opt.perturb
		self.kl = kl

	def forward(self, img1_1, img1_2, img2_1=None, img2_2=None, transformed_coordinates=None):
		device = img1_1.device
		loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
		pos_dist = 0.
		neg_dist = 0.
		distill_loss_all = 0.
		has_grad = False

		n_valid_samples = 0
		batch_size = img1_1.size(0)
		# print(img1_1.size())

		for idx_in_batch in range(batch_size):
			# Network output
			# shape: [c, h1, w1]
			dense_features1 = img1_1[idx_in_batch]
			c, h1, w1 = dense_features1.size()  # [256, 48, 48]

			# shape: [c, h2, w2]
			dense_features2 = img1_2[idx_in_batch]
			_, h2, w2 = dense_features2.size()  # [256, 48, 48]

			# shape: [c, h1 * w1]
			all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
			descriptors1 = all_descriptors1

			# Warp the positions from image 1 to image 2\
			# shape: [2, h1 * w1], coordinate in [h1, w1] dim,
			# dim 0: y, dim 1: x, positions in feature map
			fmap_pos1 = grid_positions(h1, w1, device)
			# shape: [2, h1 * w1], coordinate in image level (4 * h1, 4 * w1)
			# pos1 = upscale_positions(fmap_pos1, scaling_steps=self.scaling_steps)
			pos1 = fmap_pos1
			pos1, pos2, ids = warp(pos1, h1, w1, 
				transformed_coordinates[idx_in_batch], self.perturb)

			# print(descriptors1.shape, dense_features2.shape, transformed_coordinates.shape)
			# print(pos1.shape, pos2.shape, ids.shape)
			# print(pos1, '====', pos2, '====', ids)
			# exit()
			# shape: [2, num_ids]
			fmap_pos1 = fmap_pos1[:, ids]
			# shape: [c, num_ids]
			descriptors1 = descriptors1[:, ids]

			# Skip the pair if not enough GT correspondences are available
			if ids.size(0) < 128:
				continue

			# Descriptors at the corresponding positions
			# fmap_pos2 = torch.round(downscale_positions(pos2, \
			# 	scaling_steps=self.scaling_steps)).long()  # [2, hw]
			fmap_pos2 = torch.round(pos2).long()  # [2, hw]

			# [256, 48, 48] -> [256, hw]
			descriptors2 = F.normalize(
				dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)
			
			# [hw, 1, 256] @ [hw, 256, 1] -> [hw, hw]
			positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(1) @ \
				descriptors2.t().unsqueeze(2)).squeeze()  
				
			position_distance = torch.max(torch.abs(fmap_pos2.unsqueeze(2).float() - 
				fmap_pos2.unsqueeze(1)), dim=0)[0]  # [hw, hw]
			# print(position_distance)
			is_out_of_safe_radius = position_distance > self.safe_radius
			distance_matrix = 2 - 2 * (descriptors1.t() @ descriptors2)  # [hw, hw]
			negative_distance2 = torch.min(distance_matrix + (1 - 
				is_out_of_safe_radius.float()) * 10., dim=1)[0]  # [hw]

			all_fmap_pos1 = grid_positions(h1, w1, device)
			position_distance = torch.max(torch.abs(fmap_pos1.unsqueeze(2).float() - 
				all_fmap_pos1.unsqueeze(1)), dim=0)[0]
			# print(position_distance)
			is_out_of_safe_radius = position_distance > self.safe_radius
			distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
			negative_distance1 = torch.min(distance_matrix + (1 - 
				is_out_of_safe_radius.float()) * 10., dim=1)[0]

			# print(distance_matrix.shape, negative_distance1.shape)
			diff = positive_distance - torch.min(negative_distance1, negative_distance2)
			# diff = diff * 5.

			if not self.kl:
				loss = loss + torch.mean(F.relu(self.margin + diff))
			else:
				# distillation loss
				# student model correlation
				student_distance = torch.matmul(descriptors1.transpose(0, 1), descriptors2)
				student_distance = student_distance / self.temperature
				student_distance = F.log_softmax(student_distance, dim=1)

				# teacher model correlation
				teacher_dense_features1 = img2_1[idx_in_batch]
				c, h1, w1 = dense_features1.size()
				teacher_descriptors1 = F.normalize(teacher_dense_features1.view(c, -1), dim=0)
				teacher_descriptors1 = teacher_descriptors1[:, ids]

				teacher_dense_features2 = img2_2[idx_in_batch]
				teacher_descriptors2 = F.normalize(
					teacher_dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)
				
				teacher_distance = torch.matmul(
					teacher_descriptors1.transpose(0, 1), teacher_descriptors2)
				teacher_distance = teacher_distance / self.temperature
				teacher_distance = F.softmax(teacher_distance, dim=1)

				distill_loss = F.kl_div(student_distance, teacher_distance, \
					reduction='batchmean') * self.distill_weight
				distill_loss_all += distill_loss

				loss = loss + torch.mean(F.relu(self.margin + diff)) + distill_loss

			pos_dist = pos_dist + torch.mean(positive_distance)
			neg_dist = neg_dist + torch.mean(torch.min(negative_distance1, negative_distance2))

			has_grad = True
			n_valid_samples += 1
		
		if not has_grad:
			raise NotImplementedError

		loss = loss / n_valid_samples
		pos_dist = pos_dist / n_valid_samples
		neg_dist = neg_dist / n_valid_samples

		if not self.kl:
			return loss, pos_dist, neg_dist
		else:
			distill_loss_all = distill_loss_all / n_valid_samples
			return loss, pos_dist, neg_dist, distill_loss_all

class GANLoss(nn.Module):
	def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode = gan_mode
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss()
		elif gan_mode in ['wgangp']:
			self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def get_target_tensor(self, prediction, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)

	def forward(self, prediction, target_is_real):
		if self.gan_mode in ['lsgan', 'vanilla']:
			target_tensor = self.get_target_tensor(prediction, target_is_real)
			loss = self.loss(prediction, target_tensor)
		elif self.gan_mode == 'wgangp':
			if target_is_real:
				loss = -prediction.mean()
			else:
				loss = prediction.mean()
		return loss

class SWD(nn.Module):
	def __init__(self):
		super(SWD, self).__init__()
		self.l1loss = torch.nn.L1Loss() 

	def forward(self, fake_samples, true_samples, k=0):
		N, C, H, W = true_samples.shape

		num_projections = C//2

		true_samples = true_samples.view(N, C, -1)
		fake_samples = fake_samples.view(N, C, -1)

		projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
		projections = torch.FloatTensor(projections).to(true_samples.device)
		projections = F.normalize(projections, p=2, dim=1)

		projected_true = projections @ true_samples
		projected_fake = projections @ fake_samples

		sorted_true, true_index = torch.sort(projected_true, dim=2)
		sorted_fake, fake_index = torch.sort(projected_fake, dim=2)
		return self.l1loss(sorted_true, sorted_fake).mean() 

class MultiLoss(nn.Module):
	def __init__(self):
		super(MultiLoss, self).__init__()
		self.l1 = L1Loss()
		self.swd = SWDLoss()
	
	def forward(self, sr, hr):
		loss_EDSR_L1 = 0
		loss_EDSR_SWD = 0
		for scale in [0.5, 1, 2, 4]:
			data_sr = nn.functional.interpolate(input=sr, scale_factor=scale/4, mode='bilinear', align_corners=True)
			data_hr = nn.functional.interpolate(input=hr, scale_factor=scale/4, mode='bilinear', align_corners=True)

			loss_EDSR_L1 = loss_EDSR_L1 + self.l1(data_sr, data_hr).mean() * scale
			loss_EDSR_SWD = loss_EDSR_SWD + self.swd(data_sr, data_hr).mean() * scale

		loss_EDSR_L1 = loss_EDSR_L1.mean() / 7.5
		loss_EDSR_SWD = loss_EDSR_SWD.mean() / 7.5
		return loss_EDSR_L1, loss_EDSR_SWD, loss_EDSR_L1 + loss_EDSR_SWD


from .cobiloss.cobiloss import CX_loss, symetric_CX_loss

class CobiLoss(nn.Module):
	def __init__(self):
		super(CobiLoss,self).__init__()

	def forward(self, T_features, I_features):
		# return CX_loss(T_features, I_features)
		N, C, _, _ = T_features.size()
		kernel = 16

		T_features = F.unfold(T_features, kernel_size=(kernel, kernel), padding=0, stride=1) # [N, 27, 4096] [N, C*k*k, H*W]
		I_features = F.unfold(I_features, kernel_size=(kernel, kernel), padding=0, stride=1) # [N, 27, 4096] [N, C*k*k, H*W]
		
		p = I_features.shape[2]
		
		T_features = T_features.view(N, C, kernel, kernel, p).permute(0, 4, 1, 2, 3).contiguous()
		T_features = T_features.view(N, p*C, kernel, kernel)

		I_features = I_features.view(N, C, kernel, kernel, p).permute(0, 4, 1, 2, 3).contiguous()
		I_features = I_features.view(N, p*C, kernel, kernel)

		# true_samples = T_features.view(N, C, kernel, kernel, p).permute(0, 4, 1, 2, 3).contiguous()
		# true_samples = true_samples.view(N, p, C, kernel*kernel)
		# fake_samples = I_features.view(N, C, kernel, kernel, p).permute(0, 4, 1, 2, 3).contiguous()
		# fake_samples = fake_samples.view(N, p, C, kernel*kernel)

		# projections = torch.from_numpy(np.random.normal(size=(C, C)).astype(np.float32))
		# projections = torch.FloatTensor(projections).to(true_samples.device)
		# projections = F.normalize(projections, p=2, dim=1)

		# projected_true = projections @ true_samples   
		# projected_fake = projections @ fake_samples 

		# sorted_true, true_index = torch.sort(projected_true, dim=3)
		# sorted_fake, fake_index = torch.sort(projected_fake, dim=3)

		# sorted_true, true_index = torch.sort(true_samples, dim=3)
		# sorted_fake, fake_index = torch.sort(fake_samples, dim=3)

		# T_features = sorted_true.view(N, p*C, kernel, kernel)
		# I_features = sorted_fake.view(N, p*C, kernel, kernel)

		return CX_loss(T_features, I_features) 

class TVLoss(nn.Module):
	def __init__(self, TVLoss_weight=1):
		super(TVLoss,self).__init__()
		self.TVLoss_weight = TVLoss_weight

	def forward(self, x):
		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h = self._tensor_size(x[:,:,1:,:])
		count_w = self._tensor_size(x[:,:,:,1:])
		h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
		w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
		return self.TVLoss_weight * 2 * (h_tv/count_h+w_tv/count_w) / batch_size

	def _tensor_size(self,t):
		return t.size()[1] * t.size()[2] * t.size()[3]

class FilterLoss(nn.Module): # kernel_size%2=1
	def __init__(self):
		super(FilterLoss, self).__init__()

	def forward(self, filter_weight):  # [out, in, kernel_size, kernel_size]
		weight = filter_weight
		out_c, in_c, k, k = weight.shape 
		index = torch.arange(-(k//2), k//2+1, 1)
		# print(index)
		index = index.to(filter_weight.device)
		index = index.unsqueeze(dim=0).unsqueeze(dim=0)  # [1, 1, kernel_size] 
		index_i = index.unsqueeze(dim=3)  # [1, 1, kernel_size, 1]  
		index_j = index.unsqueeze(dim=0)  # [1, 1, 1, kernel_size]  

		diff = torch.mean(weight*index_i, dim=2).abs() + torch.mean(weight*index_j, dim=3).abs()
		return diff.mean()






LOSS_TYPES = ['cosine']
def contextual_loss(x=torch.Tensor,
                    y=torch.Tensor,
                    band_width= 0.5,
                    loss_type= 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """
    #print('band_width:',band_width)
    #assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
 
    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)

    r_m = torch.max(cx, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / 0.5) , 1, r_m[1])
    cx = torch.sum(torch.squeeze(r_m[0]*c,1), dim=1)/ torch.sum(torch.squeeze(c,1), dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5)) 

    return cx_loss




def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx

def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist
	
from collections import namedtuple
class VGG19_(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19_, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2',
                           'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_4, h_relu4_4, h_relu5_4)

        return out

class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.
    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width = 0.5,
                 loss_type = 'cosine',
                 use_vgg = True,
                 vgg_layer = 'relu3_4'):

        super(ContextualLoss, self).__init__()


        self.band_width = band_width

        if use_vgg:
            print('use_vgg:',use_vgg)
            self.vgg_model = VGG19_()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'
            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        return contextual_loss(x, y, self.band_width)
