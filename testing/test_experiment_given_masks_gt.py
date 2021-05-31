import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn
# from floorplan_dataset_maps_functional import FloorplanGraphDataset, floorplan_collate_fn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw, ImageFont
from reconstruct import reconstructFloorplan
import svgwrite
from utils import ID_COLOR
from models_exp_high_res import Generator
# from models_exp_3 import Generator


from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_variations", type=int, default=1, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exps', help="destination folder")

opt = parser.parse_args()
print(opt)

target_set = 'D'
phase='eval'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/gen_housegan_E_1000000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/functional_graph_fixed_A_300000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_functional_graph_with_l1_loss_attempt_3_A_550000.pth'
checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_128_A_750000.pth'

PREFIX = "/home/nelson/Workspace/autodesk/housegan2/"
IM_SIZE = 128


def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(g_true[0]):
        _type = label+1 
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)], color='b',weight=4)    
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')

    edges = G_true.edges()
    colors = ['black' for u,v in edges]
    weights = [4 for u,v in edges]

    nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=14, font_weight='bold', edges=edges, edge_color=colors, width=weights, with_labels=True)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return rgb_arr

        
import cv2
import webcolors

def draw_masks(masks, real_nodes):

#     transp = Image.new('RGBA', img.size, (0,0,0,0))  # Temp drawing image.
#     draw = ImageDraw.Draw(transp, "RGBA")
#     draw.ellipse(xy, **kwargs)
#     # Alpha composite two images together and replace first with result.
#     img.paste(Image.alpha_composite(img, transp))
    
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))  # Semitransparent background.
    for m, nd in zip(masks, real_nodes):
        reg = Image.new('RGBA', (IM_SIZE, IM_SIZE), (0,0,0,0))
        dr_reg = ImageDraw.Draw(reg)
        m[m>0] = 255
        m[m<0] = 0
        m = m.detach().cpu().numpy()
        m = Image.fromarray(m)
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        # r, g, b = random.randint(125, 255), random.randint(125, 255), random.randint(125, 255)
        dr_reg.bitmap((0, 0), m.convert('L'), fill=(r, g, b, 64))
        reg = reg.resize((256, 256))
        
        bg_img.paste(Image.alpha_composite(bg_img, reg))

  
    for m, nd, k in zip(masks, real_nodes, range(len(masks))):
        cnt = Image.new('RGBA', (256, 256), (0,0,0,0))
        dr_cnt = ImageDraw.Draw(cnt)
        mask = np.zeros((256,256,3)).astype('uint8')
        m[m>0] = 255
        m[m<0] = 0
        m = m.detach().cpu().numpy()[:, :, np.newaxis].astype('uint8')
        m = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 
        ret,thresh = cv2.threshold(m,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
            contours = [c for c in contours]
        for c in contours:
            c = c[:, 0, :]
            ind = np.lexsort((c[:,0], c[:,1]))
            x, y = c[ind, :][0]+1
            fnt = ImageFont.truetype("arial.ttf", 14)
            dr_cnt.text((x, y), str(k), font=fnt, fill=(255,0,0,256))

        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 1)
        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert('L'), fill=(r, g, b, 256))
        bg_img.paste(Image.alpha_composite(bg_img, cnt))

    return bg_img

def draw_floorplan(dwg, junctions, juncs_on, lines_on):

    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])
        x2, y2 = np.array(junctions[l])
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=1.0))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])
        dwg.add(dwg.circle(center=(float(x), float(y)), r=3, stroke='red', fill='white', stroke_width=2, opacity=1.0))
    return 

# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load(checkpoint))

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '../'

# Initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#  Vectorize
# ------------
globalIndex = 0
final_images = []
target_graph = list([300])
page_count = 0
n_rows = 0
feats_tensor = []
masks_tensor = []

runs = glob.glob('./runs/run_*.png')
run_n = len(runs)
print("run {} ...".format(run_n))
# fixed_nodes_state = []
fixed_nodes_state = [0, 1, 2, 3, 4, 5, 6, 7]
# fixed_nodes_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i, batch in enumerate(fp_loader):
    if i not in target_graph:
        continue
        
    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch

    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    for k in range(opt.num_variations):

        if run_n == 0:
            fixed_nodes_state = []

        # Generate a batch of images
        ind_fixed_nodes = torch.tensor(fixed_nodes_state)
        ind_not_fixed_nodes = torch.tensor([k for k in range(real_mks.shape[0]) if k not in ind_fixed_nodes])
        
        # Initialize given masks
        given_masks = torch.zeros_like(real_mks)
        given_masks = given_masks.unsqueeze(1)
        given_masks[ind_not_fixed_nodes.long()] = -1.0
        inds_masks = torch.zeros_like(given_masks)
        given_masks_in = torch.cat([given_masks, inds_masks], 1)

        z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
        with torch.no_grad():

            # look for given feats
            if run_n > 0:
                print('running state: {}'.format(str(fixed_nodes_state)))
                prev_mks = np.load('{}/runs/feat_run_{}.npy'.format(PREFIX, run_n-1), allow_pickle=True)
                prev_mks = torch.tensor(prev_mks).cuda().float()   
                given_masks_in[ind_fixed_nodes.long(), 0, :, :] = prev_mks[ind_fixed_nodes.long()]
                given_masks_in[ind_fixed_nodes.long(), 1, :, :] = 1.0
                curr_gen_mks = generator(z, None, given_masks_in, given_nds, given_eds)
            else:
                print('running initial state')
                curr_gen_mks = generator(z, None, given_masks_in, given_nds, given_eds)

            real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
            masks_tensor.append(curr_gen_mks)
            if run_n == 0:     

                # draw real graph and groundtruth
                graph = [real_nodes, None]
                graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
                graph_im.save('{}/runs/graph.png'.format(PREFIX))
                real_im = draw_masks(real_mks, real_nodes)
                real_im.save('{}/runs/gt.png'.format(PREFIX))
           
            # reconstruct
            fake_im = draw_masks(curr_gen_mks.clone(), real_nodes)
            fake_im.save('{}/runs/run_{}.png'.format(PREFIX, run_n))

            # save current features
            curr_gen_mks = curr_gen_mks.detach().cpu().numpy()
            print(min(list(set(list(curr_gen_mks.ravel())))), max(list(set(list(curr_gen_mks.ravel())))))    

            np.save('{}/runs/feat_run_{}.npy'.format(PREFIX, run_n), curr_gen_mks)
            exit(0)

# save data
# feats_tensor = torch.cat(feats_tensor, 0)
# masks_tensor = torch.cat(masks_tensor, 0)
# data = {'feats_tensor':feats_tensor, 'masks_tensor':masks_tensor}
# np.save('./autoencoder_data.npy', data)

# print(feats_tensor.shape, masks_tensor.shape)