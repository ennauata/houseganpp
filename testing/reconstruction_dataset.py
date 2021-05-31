#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, random, math
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import math
import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import random
from utils import mask_to_bb, ROOM_CLASS, ID_COLOR

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False):
		block = []
		
		if upsample:
				if spec_norm:
						block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels, \
																									 kernel_size=k, stride=s, \
																									 padding=p, bias=True)))
				else:
						block.append(torch.nn.ConvTranspose2d(in_channels, out_channels, \
																									 kernel_size=k, stride=s, \
																									 padding=p, bias=True))
		else:
				if spec_norm:
						block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels, \
																											 kernel_size=k, stride=s, \
																											 padding=p, bias=True)))
				else:        
						block.append(torch.nn.Conv2d(in_channels, out_channels, \
																											 kernel_size=k, stride=s, \
																											 padding=p, bias=True))
		if "leaky" in act:
				block.append(torch.nn.LeakyReLU(0.1, inplace=True))
		elif "relu" in act:
				block.append(torch.nn.ReLU(True))
		elif "tanh":
				block.append(torch.nn.Tanh())
		elif "None":
			continue

		return block

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.enc = nn.Sequential(
			*conv_block(1, 256, 3, 2, 1, act="relu"),
			*conv_block(256, 256, 3, 2, 1, act="relu"),
			*conv_block(256, 128, 3, 1, 1, act="relu"),
			*conv_block(128, 128, 3, 1, 1, act="relu"),
			*conv_block(128, 16, 3, 1, 1, act="None"))


	def forward(self, x):
		x = x.cuda()
		x = self.enc(x.unsqueeze(1))
		return x
				

class AutoencoderDataset(Dataset):
	def __init__(self, transform=None): 
		super(Dataset, self).__init__()
		self.data = np.load('./reconstruction_data.npy', allow_pickle=True)
		self.data = dict(self.data[()])
		self.feats = self.data['feats_list']
		self.masks = self.data['gen_masks_list']
		self.nodes = self.data['nodes_list'] 
		self.edges = self.data['edges_list']

		self.transform = transform
	def __len__(self):
		return len(self.feats)

	def __getitem__(self, index):
		feat = self.feats[index]
		mask = self.masks[index]
		nodes = self.nodes[index]
		edges = self.edges[index]

		# mask[mask>0] = 1.0
		# mask[mask<=0] = -1.0

		# mask = self.transform(mask.unsqueeze(0)).squeeze(0)
		# print(mask.shape, feat.shape)

		return feat, mask, nodes, edges

