import random
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from .imlib import imlib
from util.util import *
import torch
import colour_demosaicing

class ETHJOINTDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='ethjoint'):
		super(ETHJOINTDataset, self).__init__(opt, split, dataset_name)
		if self.root == '':
			rootlist = ['/data/dataset/Zurich-RAW-to-DSLR',
						# '/opt/data/private/Zurich-RAW-to-DSLR',
						'/opt/data/common/Datasets/Zurich-RAW-to-DSLR']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break

		self.batch_size = opt.batch_size
		self.mode = opt.mode  # RGB, Y or L=
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.imio_raw = imlib('RAW', fmt='HWC', lib='cv2')

		# self.patch_size = 192
		# self.patch_size_lr = self.patch_size // 2

		if split == 'train':
			self.raw_dir = os.path.join(self.root, 'train', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'train', 'canon')
			self.names = ['%s'%i for i in range(0, 46839)]  #  19, 20  0, 46839
			self._getitem = self._getitem_train

		elif split == 'val':
			self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'test', 'canon')
			self.names = ['%s'%i for i in range(0, 1204)] #1204
			self._getitem = self._getitem_test

		elif split == 'test':
			self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'test', 'canon')
			self.names = ['%s'%i for i in range(0, 1204)]  #  19, 20  0, 46839
			self._getitem = self._getitem_test

		elif split == 'visual':
			self.raw_dir = os.path.join(self.root, 'full_resolution/huawei_raw')
			self.names = ['1072', '1096', '1167']
			self._getitem = self._getitem_visual

		else:
			raise ValueError

		self.len_data = len(self.names)

		self.raw_images = [0] * self.len_data
		self.coord = get_coord(H=448, W=448, x=1, y=1)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
		
		dslr_image = self.imio.read(os.path.join(self.dslr_dir, self.names[idx] + ".jpg"))
		dslr_image = np.float32(dslr_image) / 255.0

		# raw_combined, raw_demosaic, dslr_image, coord = self._crop(
		# 	raw_combined, raw_demosaic, dslr_image, self.coord)

		raw_combined, raw_demosaic, dslr_image, coord = augment(
			raw_combined, raw_demosaic, dslr_image, self.coord)

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': coord,
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
		
		dslr_image = self.imio.read(os.path.join(self.dslr_dir, self.names[idx] + ".jpg"))
		dslr_image = np.float32(dslr_image) / 255.0

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': self.coord,
				'fname': self.names[idx]}

	def _getitem_visual(self, idx):
		raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
		h, w = raw_demosaic.shape[-2:]
		coord = get_coord(H=h, W=w, x=1, y=1)

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': raw_combined,
				'coord': coord,
				'fname': self.names[idx]}

	def _process_raw(self, raw):
		raw = remove_black_level(raw)
		raw_combined = extract_bayer_channels(raw)
		raw_demosaic = get_raw_demosaic(raw)
		return raw_combined, raw_demosaic
	
	# def _crop(self, raw, raw_de, dslr, index):
	# 	ih, iw = raw.shape[-2:]
	# 	ix = random.randrange(0, iw - self.patch_size_lr + 1)
	# 	iy = random.randrange(0, ih - self.patch_size_lr + 1)
	# 	tx, ty = 2 * ix, 2 * iy
	# 	return raw[..., iy:iy+self.patch_size_lr, ix:ix+self.patch_size_lr], \
	# 		   raw_de[..., ty:ty+self.patch_size, tx:tx+self.patch_size], \
	# 		   dslr[..., ty:ty+self.patch_size, tx:tx+self.patch_size], \
	# 		   index[..., ty:ty+self.patch_size, tx:tx+self.patch_size]

	# def _extract_bayer_channels(self, raw):
	# 	raw = np.maximum(raw.astype(np.float32)-63, 0) / (4 * 255-63)
	# 	raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern='RGGB')
	# 	raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32).transpose((2, 0, 1)))
		
	# 	ch_B  = raw[1::2, 1::2]
	# 	ch_Gb = raw[0::2, 1::2]
	# 	ch_R  = raw[0::2, 0::2]
	# 	ch_Gr = raw[1::2, 0::2]
	# 	raw_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
	# 	raw_combined = np.ascontiguousarray(raw_combined.transpose((2, 0, 1)))
		
	# 	return raw_combined, raw_demosaic

	# def _prede(self, raw):
	# 	raw_image = np.expand_dims(raw, 0)  # 1 * H * W
	# 	raw_image = raw_image.astype(np.float32) / (4 * 255)
	# 	return np.ascontiguousarray(raw_image)

	# def _augment_func(self, img, hflip, vflip, rot90):
	# 	if hflip:   img = img[:, :, ::-1]
	# 	if vflip:   img = img[:, ::-1, :]
	# 	if rot90:   img = img.transpose(0, 2, 1) # CHW
	# 	return np.ascontiguousarray(img)

	# def _augment(self, *imgs):
	# 	hflip = random.random() < 0.5
	# 	vflip = random.random() < 0.5
	# 	rot90 = random.random() < 0.5
	# 	return (self._augment_func(img, hflip, vflip, rot90) for img in imgs)

	# def _get_index(self, H, W):
	# 	x_index = np.linspace(-1.0 + (1.0 / H), 1.0 - (1.0 / H), H)
	# 	x_index = np.expand_dims(x_index, axis=1)
	# 	x_index = np.tile(x_index, (1, W))
	# 	x_index = np.expand_dims(x_index, axis=0)
	# 	# print(x_index)
	# 	# print(x_index.shape)
	# 	y_index = np.linspace(-1.0 + (1.0 / W), 1.0 - (1.0 / W), W)
	# 	y_index = np.expand_dims(y_index, axis=0)
	# 	y_index = np.tile(y_index, (H, 1))
	# 	y_index = np.expand_dims(y_index, axis=0)
	# 	# print(y_index)
	# 	# print(y_index.shape)
	# 	index = np.ascontiguousarray(np.concatenate([x_index, y_index]))
	# 	index = np.float32(index)
	# 	# print(index.shape)
	# 	return index

def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	i, obj = arg
	for _ in range(3):
		try:
			obj.raw_images[i] = obj.imio_raw.read(os.path.join(obj.raw_dir, obj.names[i] + '.png'))
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	from multiprocessing.dummy import Pool
	from tqdm import tqdm
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(obj.len_data, obj)), total=obj.len_data):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass