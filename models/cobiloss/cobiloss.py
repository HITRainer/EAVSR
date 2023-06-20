import torch
import torch.nn as nn
import numpy as np
# import sklearn.manifold.t_sne
import random
import torch.nn.functional as F

class TensorAxis:
	N = 0
	H = 1
	W = 2
	C = 3


class CSFlow:
	def __init__(self, sigma=float(0.1), b=float(1.0)):
		self.b = b
		self.sigma = sigma

	def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
		self.scaled_distances = scaled_distances
		self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
		self.cs_NHWC = self.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
		# self.cs_weights_before_normalization

	@staticmethod
	def create_using_L2(I_features, T_features, sigma=float(0.1), b=float(1.0)):
		cs_flow = CSFlow(sigma, b)
		sT = T_features.shape
		sI = I_features.shape

		Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
		Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
		
		# projections = torch.from_numpy(np.random.normal(size=(sT[3], sT[3])).astype(np.float32))
		# projections = torch.FloatTensor(projections).to(Ivecs.device)
		# projections = F.normalize(projections, p=2, dim=0)

		# Ivecs = Ivecs @ projections   
		# Tvecs = Tvecs @ projections 

		Ivecs, true_index = torch.sort(Ivecs, dim=1)
		Tvecs, fake_index = torch.sort(Tvecs, dim=1)

		r_Ts = torch.sum(Tvecs * Tvecs, 2)
		r_Is = torch.sum(Ivecs * Ivecs, 2)
		raw_distances_list = []
		for i in range(sT[0]):
			Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
			A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
			cs_flow.A = A
			# print(sT, Ivecs.shape, r_Is.shape, A.shape)
			# A = tf.matmul(Tvec, tf.transpose(Ivec))
			r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
			dist = r_T - 2 * A + r_I
			dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
			# protecting against numerical problems, dist should be positive
			dist = torch.clamp(dist, min=float(0.0))
			# dist = tf.sqrt(dist)
			raw_distances_list += [dist]

		cs_flow.raw_distances = torch.cat(raw_distances_list)

		relative_dist = cs_flow.calc_relative_distances()
		cs_flow.__calculate_CS(relative_dist)
		return cs_flow

	# --
	@staticmethod
	def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
		cs_flow = CSFlow(sigma, b)
		sT = T_features.shape
		sI = I_features.shape

		Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
		Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
		raw_distances_list = []
		for i in range(sT[0]):
			Ivec, Tvec = Ivecs[i], Tvecs[i]
			dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
			dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
			# protecting against numerical problems, dist should be positive
			dist = torch.clamp(dist, min=float(0.0))
			# dist = tf.sqrt(dist)
			raw_distances_list += [dist]

		cs_flow.raw_distances = torch.cat(raw_distances_list)

		relative_dist = cs_flow.calc_relative_distances()
		cs_flow.__calculate_CS(relative_dist)
		return cs_flow

	# --
	@staticmethod
	def create_using_dotP(I_features, T_features, sigma=float(1), b=float(1.0)):
		cs_flow = CSFlow(sigma, b)
		# prepare feature before calculating cosine distance
		T_features, I_features = cs_flow.center_by_T(T_features, I_features)
		T_features = CSFlow.l2_normalize_channelwise(T_features)
		I_features = CSFlow.l2_normalize_channelwise(I_features)

		# work seperatly for each example in dim 1
		cosine_dist_l = []
		N = T_features.size()[0]
		for i in range(N):
			T_features_i = T_features[i, :, :, :].unsqueeze_(0)  # 1HWC --> 1CHW
			I_features_i = I_features[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
			patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
			cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
			cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
			cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

		cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

		cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

		relative_dist = cs_flow.calc_relative_distances()
		cs_flow.__calculate_CS(relative_dist)
		return cs_flow

	def calc_relative_distances(self, axis=TensorAxis.C):
		epsilon = 1e-5
		div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
		relative_dist = self.raw_distances / (div + epsilon)
		return relative_dist

	@staticmethod
	def sum_normalize(cs, axis=TensorAxis.C):
		reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
		cs_normalize = torch.div(cs, reduce_sum)
		return cs_normalize

	def center_by_T(self, T_features, I_features):
		# assuming both input are of the same size
		# calculate stas over [batch, height, width], expecting 1x1xDepth tensor
		axes = [0, 1, 2]
		self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
		self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
		self.T_features_centered = T_features - self.meanT
		self.I_features_centered = I_features - self.meanT

		return self.T_features_centered, self.I_features_centered

	@staticmethod
	def l2_normalize_channelwise(features):
		norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
		features = features.div(norms)
		return features

	def patch_decomposition(self, T_features):
		# 1HWC --> 11PC --> PC11, with P=H*W
		(N, H, W, C) = T_features.shape
		P = H * W
		patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
		return patches_PC11

	@staticmethod
	def pdist2(x, keepdim=False):
		sx = x.shape
		x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
		differences = x.unsqueeze(2) - x.unsqueeze(1)
		distances = torch.sum(differences**2, -1)
		if keepdim:
			distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
		return distances

	@staticmethod
	def calcR_static(sT, order='C', deformation_sigma=0.05):
		# oreder can be C or F (matlab order)
		pixel_count = sT[0] * sT[1]

		rangeRows = range(0, sT[1])
		rangeCols = range(0, sT[0])
		Js, Is = np.meshgrid(rangeRows, rangeCols)
		row_diff_from_first_row = Is
		col_diff_from_first_col = Js

		row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
		col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

		rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
		colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
		R = rowDiffs ** 2 + colDiffs ** 2
		R = R.astype(np.float32)
		R = np.exp(-(R) / (2 * deformation_sigma ** 2))
		return R


def random_sampling(tensor_NHWC, n, indices=None):
	N, H, W, C = tensor_NHWC.size()
	S = H * W
	tensor_NSC = torch.reshape(tensor_NHWC, [N, S, C])

	if indices is None:
		all_indices = list(range(S))
		random.shuffle(all_indices)
		shuffled_indices = torch.from_numpy(np.array(all_indices)).type(torch.int64).to(tensor_NHWC.device)

		no_shuffled_indices = torch.from_numpy(np.array(list(range(n)))).type(torch.int64).to(tensor_NHWC.device)
		indices = shuffled_indices[no_shuffled_indices] if indices is None else indices
		# torch.gather(shuffled_indices, 0, no_shuffled_indices) if indices is None else indices
	res = tensor_NSC[:, indices, :] # torch.index_select(tensor_NSC, 1, indices)
	# print(shuffled_indices.shape, no_shuffled_indices.shape, tensor_NSC.shape, indices.shape)

	return res, indices

def random_pooling(feats, output_1d_size=100):
	N, H, W, C = feats[0].size()
	# print(feats[0].size())
	feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
	res = [feats_sampled_0]
	for i in range(1, len(feats)):
		feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
		res.append(feats_sampled_i)

	res = [torch.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]

	return res

class DWTForward(nn.Conv2d):
    def __init__(self, in_channels=64):
        super(DWTForward, self).__init__(in_channels, in_channels*4, 2, 2,
                                  groups=in_channels, bias=False)
        weight = torch.tensor([[[[0.5,  0.5], [ 0.5,  0.5]]],
                               [[[0.5,  0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [ 0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5,  0.5]]]],
                              dtype=torch.get_default_dtype()
                             ).repeat(in_channels, 1, 1, 1)# / 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)

# --------------------------------------------------
#		   CX loss
# --------------------------------------------------

def CX_VGG_loss(T_features, I_features, nnsigma=0.5, b=1.0, w_spatial=0.1, maxsize=63, deformation=False, dis=False):

	# since this originally Tensorflow implemntation
	# we modify all tensors to be as TF convention and not as the convention of pytorch.
	def from_pt2tf(Tpt):
		Ttf = Tpt.permute(0, 2, 3, 1)
		return Ttf
	# N x C x H x W --> N x H x W x C
	_,_,fh,fw = T_features.size()
	if fh*fw > maxsize**2:
		
		T_features_tf, I_features_tf = random_pooling([from_pt2tf(T_features), from_pt2tf(I_features)])
	else:
		
		T_features_tf = from_pt2tf(T_features)
		I_features_tf = from_pt2tf(I_features)

	rows = torch.range(1,T_features_tf.shape[1])
	cols = torch.range(1,T_features_tf.shape[2])
	rows = rows.type(torch.float32)/(T_features_tf.shape[1]) * 255.
	cols = rows.type(torch.float32)/(T_features_tf.shape[0]) * 255.


	features_grid = torch.meshgrid(rows, cols).to(T_features.device)
	features_grid = torch.cat([torch.unsqueeze(features_grid_i, 2) for features_grid_i in features_grid], axis=2)
	features_grid = torch.unsqueeze(features_grid, axis=0)
	features_grid = features_grid.repeat([T_features_tf.shape[0], 1, 1, 1])

	cs_flow_sp = CSFlow.create_using_L2(features_grid, features_grid, nnsigma, b)

	cs_flow = CSFlow.create_using_dotP(I_features_tf, T_features_tf,nnsigma, b)

	# To:
	cs = cs_flow.cs_NHWC
	cs_sp = cs_flow_sp.cs_NHWC
	# print(cs_sp.type())
	# print(cs.type())
	cs_comb = cs * (1.-w_spatial) + cs_sp * w_spatial
	k_max_NC = torch.max(torch.max(cs_comb, 1)[0],1)[0]
	

	CS = torch.mean(k_max_NC, 1)
	CX_as_loss = 1 - CS
	CX_loss = -torch.log(1 - CX_as_loss)
	CX_loss = torch.mean(CX_loss)
	return CX_loss

def CX_loss(T_features, I_features, nnsigma=1, b=0.5, w_spatial=0.2, maxsize=101, deformation=False, dis=False):
	device = T_features.device
	dwt = DWTForward(I_features.shape[1]).to(device)
	T_features = dwt(T_features)
	I_features = dwt(I_features)
	
	# since this originally Tensorflow implemntation
	# we modify all tensors to be as TF convention and not as the convention of pytorch.
	def from_pt2tf(Tpt):
		Ttf = Tpt.permute(0, 2, 3, 1)
		return Ttf
	# N x C x H x W --> N x H x W x C

	_,_,fh,fw = T_features.size()
	if fh*fw > maxsize**2:
		T_features_tf, I_features_tf = random_pooling([from_pt2tf(T_features),from_pt2tf(I_features)])
		# print(T_features_tf.shape, I_features_tf.shape)
	else:
		T_features_tf = from_pt2tf(T_features)
		I_features_tf = from_pt2tf(I_features)
	#
	# print(T_features_tf.shape)

	rows = torch.arange(0,T_features_tf.shape[1]).to(device)
	cols = torch.arange(0,T_features_tf.shape[2]).to(device)
	rows = rows.type(torch.float32)/(T_features_tf.shape[1]) # * 255.
	cols = rows.type(torch.float32)/(T_features_tf.shape[0]) # * 255.

	features_grid = torch.meshgrid(rows, cols)
	features_grid = torch.cat([torch.unsqueeze(features_grid_i, 2) for features_grid_i in features_grid], axis=2)
	features_grid = torch.unsqueeze(features_grid, axis=0)
	features_grid = features_grid.repeat([T_features_tf.shape[0], 1, 1, 1])

	# cs_flow_sp = CSFlow.create_using_L2(features_grid, features_grid, nnsigma, b)
	# cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, nnsigma, b)

	cs_flow_sp = CSFlow.create_using_L2(features_grid, features_grid, nnsigma, b)
	cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, nnsigma, b) # [N,H,W,C]->[N,H,W,H*W]

	# To:
	cs = cs_flow.cs_NHWC
	cs_sp = cs_flow_sp.cs_NHWC
	cs_comb = cs * (1.-w_spatial) # + cs_sp * w_spatial
	
	# print(features_grid.shape, I_features_tf.shape, cs_comb.shape)

	k_max_NC = torch.max(torch.max(cs_comb, 1)[0],1)[0]
	# print(cs_comb.shape, k_max_NC.shape)

	CS = torch.mean(k_max_NC, 1)
	CX_loss = -torch.log(CS + 1e-5)
	# CX_as_loss = 1 - CS
	# CX_loss = -torch.log(1 - CX_as_loss)
	CX_loss = torch.mean(CX_loss)
	return CX_loss

def symetric_CX_loss(T_features, I_features):
	score = (CX_loss(T_features, I_features) + CX_loss(I_features, T_features)) / 2
	return score

if __name__ == '__main__':

	for i in range(1):
		img1 = torch.rand(1, 400, 24, 24).to('cuda:1')
		img2 = torch.rand(1, 400, 24, 24).to('cuda:1')
		loss = CX_loss(img1, img2)
		print(loss)