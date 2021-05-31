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

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
# import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFont, ImageColor
from pygraphviz import *
import cv2
from torchvision.utils import save_image
import networkx as nx
import copy
from misc.intersections import doIntersect
import svgwrite
import random
import matplotlib.pyplot as plt
import webcolors
cv2.setNumThreads(0)
EXP_ID = random.randint(0, 1000000)

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}
CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}


# Select random nodes
def selectRandomNodes(nd_to_sample, batch_size):
    fixed_rooms_num = []
    fixed_nodes = []
    shift = 0
    for k in range(batch_size):
        rooms = np.where(nd_to_sample == k)
        rooms_num = np.array(rooms).shape[-1]
        N = np.random.randint(rooms_num, size=1)
        fixed_nodes_state = torch.tensor(np.random.choice(list(range(rooms_num)), size=N, replace=False)).cuda() ##torch.tensor(list(range(rooms_num))).long().cuda() ##
        fixed_nodes_state += shift
        fixed_nodes.append(fixed_nodes_state)
        shift += rooms_num 
    fixed_nodes = torch.cat(fixed_nodes)
    bin_fixed_nodes = torch.zeros((nd_to_sample.shape[0], 1))
    bin_fixed_nodes[fixed_nodes] = 1.0
    bin_fixed_nodes = bin_fixed_nodes.float().cuda()
    return fixed_nodes, bin_fixed_nodes

# Select nodes per room type
def selectNodesTypes(nd_to_sample, batch_size, nds):
    all_types, fixed_rooms_num, fixed_nodes, shift = [ROOM_CLASS[k]-1 for k in ROOM_CLASS], [], [], 0
    for k in range(batch_size):
        rooms = np.where(nd_to_sample == k)
        rooms_num = np.array(rooms).shape[-1]
        _types = np.where(nds[rooms]==1)[1]
        _t = [t for t in all_types if random.uniform(0, 1) > 0.5]
        fixed_rooms = [r for r, _t_x in enumerate(_types) if _t_x in _t]
#         print(' existing types: {} \n sected types: {} \n fixed rooms {}'.format('-'.join([str(i) for i in _types]), '-'.join([str(i) for i in _t]), '-'.join([str(i) for i in fixed_rooms])))
        fixed_nodes_state = torch.tensor(fixed_rooms).cuda()
        fixed_nodes_state += shift
        fixed_nodes.append(fixed_nodes_state)
        shift += rooms_num 
    fixed_nodes = torch.cat(fixed_nodes)
    bin_fixed_nodes = torch.zeros((nd_to_sample.shape[0], 1))
    bin_fixed_nodes[fixed_nodes.long()] = 1.0
    bin_fixed_nodes = bin_fixed_nodes.float().cuda()
    return fixed_nodes, bin_fixed_nodes

def fix_nodes(prev_mks, ind_fixed_nodes):
    given_masks = torch.tensor(prev_mks)
    ind_not_fixed_nodes = torch.tensor([k for k in range(given_masks.shape[0]) if k not in ind_fixed_nodes])
    ## Set non fixed masks to -1.0
    given_masks[ind_not_fixed_nodes.long()] = -1.0
    given_masks = given_masks.unsqueeze(1)
    ## Add channel to indicate given nodes 
    inds_masks = torch.zeros_like(given_masks)
    inds_masks[ind_not_fixed_nodes.long()] = 0.0
    inds_masks[ind_fixed_nodes.long()] = 1.0
    given_masks = torch.cat([given_masks, inds_masks], 1)
    return given_masks
    
def _init_input(graph, prev_state=None, mask_size=64):
    # initialize graph
    given_nds, given_eds = graph
    given_nds = given_nds.float()
    given_eds = torch.tensor(given_eds).long()
    z = torch.randn(len(given_nds), 128).float()
    # unpack
    fixed_nodes = prev_state['fixed_nodes']
    prev_mks = torch.zeros((given_nds.shape[0], mask_size, mask_size))-1.0 if (prev_state['masks'] is None) else prev_state['masks']
    # initialize masks
    given_masks_in = fix_nodes(prev_mks, torch.tensor(fixed_nodes))
    return z, given_masks_in, given_nds, given_eds

def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGBA', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_masks(masks, real_nodes, im_size=256):
    
    bg_img = Image.new("RGBA", (im_size, im_size), (255, 255, 255, 255))  # Semitransparent background.
    for m, nd in zip(masks, real_nodes):
        
        # resize map
        m[m>0] = 255
        m[m<0] = 0
        m_lg = cv2.resize(m, (im_size, im_size), interpolation = cv2.INTER_AREA) 

        # pick color
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.hex_to_rgb(color)

        # set drawer
        dr_bkg = ImageDraw.Draw(bg_img)

        # draw region
        m_pil = Image.fromarray(m_lg)
        dr_bkg.bitmap((0, 0), m_pil.convert('L'), fill=(r, g, b, 256))

        # draw contour
        m_cv = m_lg[:, :, np.newaxis].astype('uint8')
        ret,thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(contours) > 0]
        cnt = np.zeros((256, 256, 3)).astype('uint8')
        cv2.drawContours(cnt, contours, -1, (255, 255, 255, 255), 1)
        cnt = Image.fromarray(cnt)
        dr_bkg.bitmap((0, 0), cnt.convert('L'), fill=(0, 0, 0, 255))

    return bg_img.resize((im_size, im_size))

def combine_images(samples_batch, nodes_batch, edges_batch, nd_to_sample, ed_to_sample):
    samples_batch = samples_batch.detach().cpu().numpy()
    nodes_batch = nodes_batch.detach().cpu().numpy()
    edges_batch = edges_batch.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    all_imgs = []
    shift = 0
    for b in range(batch_size):

        # split batch
        inds_nd = np.where(nd_to_sample==b)
        inds_ed = np.where(ed_to_sample==b)
        sps = samples_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        eds = edges_batch[inds_ed]

        # draw
        # graph_image = draw_graph_with_types(nds, eds, shift)
        _image = draw_masks(sps, nds)
        
        # store
        # all_imgs.append(torch.FloatTensor(np.array(graph_image.convert('RGBA')).\
        #                              astype('float').\
        #                              transpose(2, 0, 1))/255.0)
        all_imgs.append(torch.FloatTensor(np.array(_image.convert('RGBA')).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
        shift += len(nds)
    return torch.stack(all_imgs)

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    
    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label+1 
        if _type >= 0 and _type not in [15, 17]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
            
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    
    # add edges
    for k, m, l in g_true[1]:
        _type_k = g_true[0][k]+1  
        _type_l = g_true[0][l]+1
        if m > 0 and (_type_k not in [15, 17] and _type_l not in [15, 17]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        elif m > 0 and (_type_k==15 or _type_l==15) and (_type_l!=17 and _type_k!=17):
            if _type_k==15:
                G_true.add_edges_from([(l, -1)])   
            elif _type_l==15:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    
#     # # visualization - debug
#     print(len(node_size))
#     print(len(colors_H))
#     print(len(linewidths))
#     print(G_true.nodes())
#     print(g_true[0])
#     print(len(edgecolors))
    

    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
    nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
            font_weight='bold', edgecolors=edgecolors, edge_color=edge_color, width=4.0, with_labels=False)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    plt.close('all')
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return G_true, rgb_im

def remove_multiple_components(masks, nodes):
    
    new_masks = []
    for mk, nd in zip(masks, nodes):
        m_cv = np.array(mk)
        m_cv[m_cv>0] = 255.0
        m_cv[m_cv<0] = 0.0
        m_cv = m_cv.astype('uint8')
        ret,thresh = cv2.threshold(m_cv, 127, 255 , 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:  
            cnt_m = np.zeros_like(m_cv)
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(cnt_m, [c], 0, (255, 255, 255), -1)
            cnt_m[cnt_m>0] = 1.0
            cnt_m[cnt_m<0] = -1.0
            new_masks.append(cnt_m)
        else:
            new_masks.append(mk)
    return new_masks

def estimate_graph(masks, nodes, G_gt):
    
    # remove multiple components
    masks = remove_multiple_components(masks, nodes)
    
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []    
    edge_labels = {}
    
    # add nodes
    for k, label in enumerate(nodes):
        _type = label+1 
        if _type >= 0 and _type not in [15, 17]:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            linewidths.append(0.0)
    
    # add outside node
    G_estimated.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    
    # add node-to-door connections
    doors_inds = np.where((nodes == 14) | (nodes == 16))[0]
    rooms_inds = np.where((nodes != 14) & (nodes != 16))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:   
                m1, m2 = masks[k], masks[l]
                m1[m1>0] = 1.0
                m1[m1<=0] = 0.0
                m2[m2>0] = 1.0
                m2[m2<=0] = 0.0
                iou = np.logical_and(m1, m2).sum()/float(np.logical_or(m1, m2).sum())
                if iou > 0 and iou < 0.2:
                    doors_rooms_map[k].append((l, iou))    

    # draw connections            
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k]+1 != 15:
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                edge_labels[(l1, l2)] = k
                G_estimated.add_edges_from([(l1, l2)])
        else:
            if len(_conn) > 0:
                l1 = _conn[0][0]
                edge_labels[(-1, l1)] = k
                G_estimated.add_edges_from([(-1, l1)])

       
    # add missed edges 
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])

    # add edges colors
    colors = []
    mistakes = 0
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')

    # visualization - debug
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
    weights = [4 for u, v in G_estimated_complete.edges()]
    nx.draw(G_estimated_complete, pos, edge_color=colors, linewidths=linewidths, edgecolors=edgecolors, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', width=weights, with_labels=False)
    
    plt.tight_layout()
    plt.savefig('./dump/_fake_graph.jpg', format="jpg")
    
    rgb_im = Image.open('./dump/_fake_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    plt.close('all')

    return mistakes, rgb_im