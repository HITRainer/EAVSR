#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 18:51:59 2021

@author: yangliu
"""
import torch
from cobiloss import CX_loss,CX_VGG_loss
from vggcobi import VGG19
import torch.nn as nn

class COBIRGBLoss(nn.Module):
	def __init__(self):
		
		super(COBIRGBLoss,self).__init__()
		self.vggmodel = VGG19()

	def extractpatch(self,input,size):
		n = input.shape[0]
		out1 = torch.nn.functional.unfold(input,(size,size))
		new = out1.transpose(out1,2,1)
		new = new.reshape(n,-1,size,size)
		return new

	def forward(self,output,gt):
		outputfeature = self.vggmodel(output)
		gtfeature = self.vggmodel(gt)
		L = []
		for i in range(len(outputfeature)):
	
			loss = CX_VGG_loss(outputfeature[i],gtfeature[i])
			L.append(loss)
		vggloss = L[0]*1.0+L[1]*1.0+L[2]*0.5
		
		outputpatches = self.extractpatch(output,10)
		gtpatches = self.extractpatch(gt,10)
		CX_loss(outputpatches,gtpatches)
vggmodel = VGG19().cuda()
img1 = torch.rand([2,3,512,512]).cuda()
img2 = torch.rand([2,3,512,512]).cuda()
cobiloss = COBIRGBLoss().cuda()
# img = torch.clamp(img,0.0,1.0)
# imgfeature1 = vggmodel(img1)
# imgfeature2 = vggmodel(img2)
print(cobiloss(img1,img2))
print(cobiloss(img2,img1))
# CSFlow.CSFlow.c
