import torch

import getopt
import math
import numpy as np
import numpy
import os
import PIL
import PIL.Image
import sys
import pickle
from functools import partial
import cv2
import torch.nn.functional as F

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

# torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
######################################################

backwarp_tenGrid = {}
backwarp_tenPartial={}
def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super().__init__()
			# end

			def forward(self, tenInput):
				tenBlue = (tenInput[:, 2:3, :, :] - 0.406) / 0.225
				tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
				tenRed = (tenInput[:, 0:1, :, :] - 0.485) / 0.229

				return torch.cat([ tenRed, tenGreen, tenBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super().__init__()

				self.netBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			# end

			def forward(self, tenInput):
				return self.netBasic(tenInput)
			# end
		# end

		self.netPreprocess = Preprocess()

		self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight 
							   in torch.load('./pwc/spy-default', map_location=lambda storage, 
							   loc: storage, pickle_module=pickle).items() })
		# self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + arguments_strModel + '.pytorch', file_name='spynet-' + arguments_strModel).items() })
	# end

	def forward(self, tenOne, tenTwo):
		tenFlow = []

		tenOne = [ self.netPreprocess(tenOne) ]
		tenTwo = [ self.netPreprocess(tenTwo) ]

		for intLevel in range(5):
			if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
				tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
				tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
			# end
		# end

		tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])

		for intLevel in range(len(tenOne)):
			tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
		# end

		return tenFlow
	# end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo, net):
	# end
	assert(tenOne.shape[2] == tenTwo.shape[2])
	assert(tenOne.shape[3] == tenTwo.shape[3])

	intWidth = tenOne.shape[3]
	intHeight = tenOne.shape[2]

	# assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	# assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	# tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
	# tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tenPreprocessedOne = torch.nn.functional.interpolate(input=tenOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = torch.nn.functional.interpolate(input=net(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow
# end

##########################################################
def get_backwarp(tenFirst, tenSecond, net, flow=None):
	tenFirst_=F.interpolate(tenFirst, scale_factor=0.5, mode='bicubic', align_corners=True)
	tenSecond_=F.interpolate(tenSecond, scale_factor=0.5, mode='bicubic', align_corners=True)
	if flow is None:
		flow = get_flow(tenFirst_, tenSecond_, net)
	flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
	tenoutput = backwarp_(tenSecond, flow) 
	tenMask = tenoutput[:, -1:, :, :]
	tenMask[tenMask > 0.999] = 1.0
	tenMask[tenMask < 1.0] = 0.0
	return tenoutput[:, :-1, :, :] * tenMask, tenMask

def get_flow(tenFirst, tenSecond, net):
		with torch.no_grad():
			flow = estimate(tenFirst, tenSecond, net) 
		return flow
def backwarp_(tenInput, tenFlow):
	index = str(tenFlow.shape) + str(tenInput.device)
	if index not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
								tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
								tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
		backwarp_tenGrid[index] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

	if index not in backwarp_tenPartial:
		backwarp_tenPartial[index] = tenFlow.new_ones([ tenFlow.shape[0], 
												1, tenFlow.shape[2], tenFlow.shape[3] ])

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
						tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[index] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, 
				grid=(backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
				mode='bilinear', padding_mode='zeros', align_corners=False)

	return tenOutput


if __name__ == '__main__':
	# tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
	# tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
	ref = cv2.imread('/data/SSD/RealVSR/HR_test/223/00028.png', -1)
	H, W, C = ref.shape
	lr = cv2.imread('/data/SSD/RealVSR/HR_test/223/00029.png', -1)
	h, w, c = lr.shape
	ref = (ref.astype(np.float32) / 255.).transpose(2, 0, 1)
	lr = (lr.astype(np.float32) / 255.).transpose(2, 0, 1)

	net = Network().to('cuda:0')
	ref = torch.from_numpy(ref).unsqueeze(0).to('cuda:0')

	lr = torch.from_numpy(lr).unsqueeze(0).to('cuda:0')

	warp, mask = get_backwarp(lr, ref, net)
	warp = (np.array(warp.cpu()).squeeze(0).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
	
	cv2.imwrite('/home/wrh/Videos/warp2.png', warp)


	# objOutput = open(arguments_strOut, 'wb')

	# numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
	# numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
	# numpy.array(tenOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)

	# objOutput.close()
# end