import random
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from .imlib import imlib
from util.util import *

class RealVSRDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='RealVSR'):
		super(RealVSRDataset, self).__init__(opt, split, dataset_name)
		if not opt.dataroot == '':
			self.root = opt.dataroot
		else:
			if self.root == '':
				rootlist = ['/data/wrh/datasets/RealVSR']
				for root in rootlist:
					if os.path.isdir(root):
						self.root = root
						break

		self.batch_size = opt.batch_size
		self.mode = opt.mode  # RGB, Y or L=
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.patch_size = opt.patch_size # 48
		self.scale = opt.scale
		self.n_frame = opt.n_frame # Num of frames per sequence, e.g. 1\3\5\7
		self.n_seq = opt.n_seq # Totally frames of a video 

		if split == 'train':
			self.train_list = np.load('./options/train_realvsr.npy')
			self.lr_dirs, self.hr_dirs, self.names = self._get_image_dir(self.train_list)
			self.len_data = len(self.names)
			self._getitem = self._getitem_train

		elif split == 'val':
			self.val_list = np.load('./options/val_realvsr.npy')
			self.lr_dirs, self.hr_dirs, self.names = self._get_image_dir(self.val_list, isTrain=False)
			self._getitem = self._getitem_val
			self.len_data = len(self.names)

		elif split == 'test':
			self.test_list = np.load('./options/test_realvsr.npy')
			self.lr_dirs, self.hr_dirs, self.names = self._get_image_dir(self.test_list, isTrain=False)
			self._getitem = self._getitem_test
			self.len_data = int(math.ceil(len(self.names) / self.n_frame))

		else:
			raise ValueError

		self.lr_images = [0] * len(self.names)
		self.hr_images = [0] * len(self.names)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		idx = idx % len(self.names)
		train_name = str(self.train_list[idx]) # name of key frame (e.g. 000_00001)
		folder, frame = train_name[:3], int(train_name[-5:])
		frame_seq = [0] * self.n_frame
		hr_seq = [0] * self.n_frame
		# front
		if frame - (self.n_frame//2) < 0:
			for i in range((self.n_frame//2)-frame):
				frame_seq[i] = np.float32(self.lr_images[idx+(self.n_frame//2)-i]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+(self.n_frame//2)-i]) / 255
			for i in range((self.n_frame//2)-frame, self.n_frame):
				frame_seq[i] = np.float32(self.lr_images[idx+i-(self.n_frame//2)]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+i-(self.n_frame//2)]) / 255
		# back
		elif frame + (self.n_frame//2) >= self.n_seq:
			for i in range((self.n_frame//2), (self.n_seq-1)-frame, -1):
				frame_seq[i+(self.n_frame//2)] = np.float32(self.lr_images[idx-i]) / 255
				hr_seq[i+(self.n_frame//2)] = np.float32(self.hr_images[idx-i]) / 255
			for i in range((self.n_frame//2)+self.n_seq-frame):
				frame_seq[i] = np.float32(self.lr_images[idx+i-(self.n_frame//2)]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+i-(self.n_frame//2)]) / 255
		
		else:
			for i in range(self.n_frame):
				frame_seq[i] = np.float32(self.lr_images[idx+i-(self.n_frame//2)]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+i-(self.n_frame//2)]) / 255

		lr_seq, hr_seq = self._crop_patch(frame_seq, hr_seq)
		lr_seq, hr_seq = augment_basic(lr_seq, hr_seq)
		return {'lr_seq': np.stack(lr_seq, axis=0), # NCHW-> N: Frame;C,H,W
				'hr_seq': np.stack(hr_seq, axis=0),
				'fname': self.names[idx]}

	def _getitem_val(self, idx):
		val_name = str(self.val_list[idx]) # name of key frame (e.g. 000_00001)
		folder, frame = val_name[:3], int(val_name[-5:])
		frame_seq = [0] * self.n_frame
		hr_seq = [0] * self.n_frame
		# front
		if frame - (self.n_frame//2) < 0:
			for i in range((self.n_frame//2)-frame):
				frame_seq[i] = np.float32(self.lr_images[idx+(self.n_frame//2)-i]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+(self.n_frame//2)-i]) / 255
			for i in range((self.n_frame//2)-frame, self.n_frame):
				frame_seq[i] = np.float32(self.lr_images[idx+i-(self.n_frame//2)]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+i-(self.n_frame//2)]) / 255
		# back
		elif frame + (self.n_frame//2) >= self.n_seq:
			for i in range((self.n_frame//2), (self.n_seq-1)-frame, -1):
				frame_seq[i+(self.n_frame//2)] = np.float32(self.lr_images[idx-i]) / 255
				hr_seq[i+(self.n_frame//2)] = np.float32(self.hr_images[idx-i]) / 255
			for i in range((self.n_frame//2)+self.n_seq-frame):
				frame_seq[i] = np.float32(self.lr_images[idx+i-(self.n_frame//2)]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+i-(self.n_frame//2)]) / 255
		
		else:
			for i in range(self.n_frame):
				frame_seq[i] = np.float32(self.lr_images[idx+i-(self.n_frame//2)]) / 255
				hr_seq[i] = np.float32(self.hr_images[idx+i-(self.n_frame//2)]) / 255
		
		lr_seq = [self._crop_center(lr, p=256) for lr in frame_seq]
		hr_seq_ = [self._crop_center(hr, p=256*self.scale) for hr in hr_seq]
		
		return {'lr_seq': np.stack(lr_seq, axis=0),
				'hr_seq': np.stack(hr_seq_, axis=0),
				'fname': self.names[frame]}

	def _getitem_test(self, idx):
		if self.n_seq % self.n_frame == 0:
			index = [i*self.n_frame for i in range(self.n_seq//self.n_frame)]
		else:
			raise ValueError()
		idx = (idx//len(index) * self.n_seq) + index[idx % len(index)]
		frame_seq = [0] * self.n_frame
		hr_seq = [0] * self.n_frame
		name = [0] * self.n_frame

		for i in range(self.n_frame):
			frame_seq[i] = np.float32(self.lr_images[idx+i]) / 255
			hr_seq[i] = np.float32(self.hr_images[idx+i]) / 255
			name[i] = self.names[idx +i]

		return {'lr_seq': np.stack(frame_seq, axis=0),
				'hr_seq': np.stack(hr_seq, axis=0),
				'fname': name}
   
	def _get_image_dir(self, datalist, isTrain=True):
		lr_dirs = []
		hr_dirs = []
		image_names = []

		for file_name in datalist:
			folder, file = str(file_name[:3]), str(file_name[-5:]) 
			if isTrain: 
				lr_dirs.append(os.path.join(self.root, 'LR', folder, file+'.png'))
				hr_dirs.append(os.path.join(self.root, 'HR', folder, file+'.png'))
			else:
				lr_dirs.append(os.path.join(self.root, 'LR_test', folder, file+'.png'))
				hr_dirs.append(os.path.join(self.root, 'HR_test', folder, file+'.png'))
			image_names.append(str(file_name)+'.png')

		return lr_dirs, hr_dirs, image_names

	def _crop_patch(self, lr_seq, hr_seq):
		ih, iw = lr_seq[0].shape[-2:]
		pw = random.randrange(0, iw - self.patch_size + 1)
		ph = random.randrange(0, ih - self.patch_size + 1)
		hpw, hph = self.scale * pw, self.scale * ph
		hr_patch_size = self.scale * self.patch_size
		
		lr_patch_seq=[lr[..., ph:ph+self.patch_size, pw:pw+self.patch_size] for lr in lr_seq]
		hr_patch_seq = [hr[..., hph:hph+hr_patch_size, hpw:hpw+hr_patch_size] for hr in hr_seq]
		return lr_patch_seq, hr_patch_seq
			   

	def _crop_center(self, img, fw=0.5, fh=0.5, p=0):
		ih, iw = img.shape[-2:]
		if p != 0:
			fw = p / iw
			fh = p / ih
		patch_h, patch_w = round(ih * fh), round(iw * fw)
		ph = ih // 2 - patch_h // 2
		pw = iw // 2 - patch_w // 2
		return img[..., ph:ph+patch_h, pw:pw+patch_w]


def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	i, obj = arg
	for _ in range(3):
		try:
			# obj.lr_images[i] = obj.imio.read(obj.lr_dirs[i])
			img = obj.imio.read(obj.lr_dirs[i]).transpose(1, 2, 0) #HWC
			h, w, _ = img.shape
			img = cv2.resize(img, (w//obj.scale, h//obj.scale), interpolation=cv2.INTER_CUBIC)
			obj.lr_images[i] = img.transpose(2, 0, 1) # CHW
			obj.hr_images[i] = obj.imio.read(obj.hr_dirs[i])
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
	for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass