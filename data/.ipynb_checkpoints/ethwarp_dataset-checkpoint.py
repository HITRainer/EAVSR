import random
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from .imlib import imlib
from .eth_dataset import read_images as read_images
from util.util import *
import cv2
import torch


class ETHWARPDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='ethwarp'):
		super(ETHWARPDataset, self).__init__(opt, split, dataset_name)
		if self.root == '':
			rootlist = ['/opt/data/private/Zurich-RAW-to-DSLR',
						'/data/dataset/Zurich-RAW-to-DSLR',
						'/opt/data/common/Datasets/Zurich-RAW-to-DSLR']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break

		self.batch_size = opt.batch_size
		self.mode = opt.mode  # RGB, Y or L=
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.imio_raw = imlib('RAW', fmt='HWC', lib='cv2')

		if split == 'train':
			self.raw_dir = os.path.join(self.root, 'train', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'train', 'canon_warp_dong')
			self.dslr_mask_dir = os.path.join(self.root, 'train', 'canon_warp_mask_dong')
			self.names = ['%s'%i for i in range(0, 4800)]  # 0, 46839
			self._getitem = self._getitem_train

		elif split == 'val':
			self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'test', 'canon_warp_dong')
			self.dslr_mask_dir = os.path.join(self.root, 'test', 'canon_warp_mask_dong')
			self.names = ['%s'%i for i in range(0, 1204, 1)]
			self._getitem = self._getitem_test

		elif split == 'test':
			self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'test', 'canon_warp_dong')
			self.dslr_mask_dir = os.path.join(self.root, 'test', 'canon_warp_mask_dong')
			self.names = ['%s'%i for i in range(0, 1204, 1)]
			self._getitem = self._getitem_test

		elif split == 'visual':
			self.raw_dir = os.path.join(self.root, 'full_resolution/huawei_raw')
			self.names = ['1072', '1096', '1167']
			self._getitem = self._getitem_visual

		else:
			raise ValueError
	
		self.len_data = len(self.names)
		self.raw_images = [0] * self.len_data
		self.coord = util.get_coord(H=448, W=448, x=1, y=1)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		raw_combined = self._process_raw(self.raw_images[idx])
		
		dslr_image = self.imio.read(os.path.join(self.dslr_dir, self.names[idx] + ".jpg"))
		dslr_image = np.float32(dslr_image) / 255.0

		dslr_mask = self.imio.read(os.path.join(self.dslr_mask_dir, self.names[idx] + ".png"))
		dslr_mask = np.int8(dslr_mask / 255)

		raw_combined, dslr_image, dslr_mask, coord = self._augment(
			        raw_combined, dslr_image, dslr_mask, self.coord)    

		return {'raw': raw_combined, 
				'dslr': dslr_image,
				'dslrmask': dslr_mask,
				'coord': coord,
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		raw_combined = self._process_raw(self.raw_images[idx])
		
		dslr_image = self.imio.read(os.path.join(self.dslr_dir, self.names[idx] + ".jpg"))
		dslr_image = np.float32(dslr_image) / 255.0

		dslr_mask = self.imio.read(os.path.join(self.dslr_mask_dir, self.names[idx] + ".png"))
		dslr_mask = np.int8(dslr_mask / 255)

		return {'raw': raw_combined, 
				'dslr': dslr_image,
				'dslrmask': dslr_mask,
				'coord': self.coord,
				'fname': self.names[idx]}
	
	def _getitem_visual(self, idx):
		raw_combined = self._process_raw(self.raw_images[idx])
		h, w = raw_combined.shape[-2:]
		coord = get_coord(H=h*2, W=w*2)
		
		return {'raw': raw_combined, 
				'dslr': raw_combined,
				'dslrmask': raw_combined,
				'coord': coord,
				'fname': self.names[idx]}

    def _process_raw(self, raw):
        raw = remove_black_level(raw)
        raw_combined = extract_bayer_channels(raw)

        return raw_combined

	# def _extract_bayer_channels(self, raw):
	# 	ch_B  = raw[1::2, 1::2]
	# 	ch_Gb = raw[0::2, 1::2]
	# 	ch_R  = raw[0::2, 0::2]
	# 	ch_Gr = raw[1::2, 0::2]
	# 	RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
	# 	RAW_norm = np.maximum(RAW_combined.astype(np.float32)-63, 0) / (4 * 255-63)
	# 	return np.ascontiguousarray(RAW_norm.transpose((2, 0, 1)))

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

	# def _get_index(self, H, W, x=448/3968, y=448/2976):
	# 	x_index = np.linspace(-x + (x / W), x - (x / W), W)
	# 	x_index = np.expand_dims(x_index, axis=0)
	# 	x_index = np.tile(x_index, (H, 1))
	# 	x_index = np.expand_dims(x_index, axis=0)
	# 	# print(x_index)
	# 	# print(x_index.shape)
	# 	y_index = np.linspace(-y + (y / H), y - (y / H), H)
	# 	y_index = np.expand_dims(y_index, axis=1)
	# 	y_index = np.tile(y_index, (1, W))
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