import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw, ImageFont
import svgwrite
from models.models import Generator
# from models.models_improved import Generator

from misc.utils import _init_input, ID_COLOR, draw_masks, draw_graph, estimate_graph
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob
import cv2
import webcolors
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_variations", type=int, default=1, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exps', help="destination folder")

opt = parser.parse_args()
print(opt)

# PARAMS
target_set = 8
phase='eval'
checkpoint = './checkpoints/exp_debug_E_165000.pth'
PREFIX = "./"
IM_SIZE = 64
output_dir = "./dump"

# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
model = Generator()
model.load_state_dict(torch.load(checkpoint), strict=False)
model = model.eval()

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    model.cuda()
rooms_path = '../'

# initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn)
# optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# run inference
def _infer(graph, model, prev_state=None):
    
    # configure input to the network
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
    # run inference model
    with torch.no_grad():
        masks = model(z.to('cuda'), given_masks_in.to('cuda'), given_nds.to('cuda'), given_eds.to('cuda'))
        masks = masks.detach().cpu().numpy()
    return masks

def main():
    globalIndex = 0
    for i, sample in enumerate(fp_loader):

        # draw real graph and groundtruth
        mks, nds, eds, _, _ = sample
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
        graph = [nds, eds]
        true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
        graph_im.save('{}/{}/graph_{}.png'.format(PREFIX, output_dir, i)) # save graph

        # add room types incrementally
        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[:k+1] for k in range(10)]
        os.makedirs('./{}/'.format(output_dir), exist_ok=True)
        _round = 0
        
        # initialize layout
        state = {'masks': None, 'fixed_nodes': []}
        masks = _infer(graph, model, state)
        im0 = draw_masks(masks.copy(), real_nodes)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 
        save_image(im0, './{}/fp_init_{}.png'.format(output_dir, i), nrow=1, normalize=False)

        # generate per room type
        for _iter, _types in enumerate(selected_types):
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                if len(_types) > 0 else np.array([]) 
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = _infer(graph, model, state)
            
        # save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
        save_image(imk, './{}/fp_final_{}.png'.format(output_dir, i), nrow=1, normalize=False)
        asdasd
        
#         miss, gim = estimate_graph(masks, real_nodes, true_graph_obj)        
#         if miss >= 0:
#             gim = torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0
#             save_image(imk, './{}/fp_graph_{}_iter_{}_final.png'.format(output_dir, _iter, _round), nrow=1, normalize=False)
#             save_image(gim, './{}/gp_graph_{}_iter_{}_final.png'.format(output_dir, _iter, _round), nrow=1, normalize=False)
#             _round += 1
#         end = time.time()

if __name__ == '__main__':
    main()