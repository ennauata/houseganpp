import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn
# from floorplan_dataset_maps_functional import FloorplanGraphDataset, floorplan_collate_fn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw, ImageFont
import svgwrite
from misc.utils import ID_COLOR, ROOM_CLASS
from models.models_exp_high_res import Generator
# from models_exp_3 import Generator

from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob
import cv2
import webcolors
from pytorch_fid.fid_score import calculate_fid_given_paths
from bayes_opt import BayesianOptimization
import itertools
from functools import partial
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--num_variations", type=int, default=1, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exps', help="destination folder")

opt = parser.parse_args()
print(opt)

target_set = 8
phase='train'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/gen_housegan_E_1000000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/functional_graph_fixed_A_300000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_functional_graph_with_l1_loss_attempt_3_A_550000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_128_A_750000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_with_doors_64x64_per_room_type_A_230000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_random_350000_A_200000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_random_node_type_A_350000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_per_room_type_enc_dec_plus_local_A_260000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_autoencoder_A_72900.pth'
checkpoint = './checkpoints/exp_random_types_attempt_3_A_500000_G.pth'

PREFIX = "./"
IM_SIZE = 64


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
        elif m > 0 and (_type_k==15 or _type_l==15):
            if _type_k==15:
                G_true.add_edges_from([(l, -1)])   
            else:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    
#     # visualization - debug
#     plt.figure()
#     pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
#     nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
#             font_weight='bold', edgecolors=edgecolors, width=4.0, with_labels=True)
    
#     plt.tight_layout()
#     plt.savefig('./dump/_true_graph.jpg', format="jpg")
#     plt.close('all')
#     rgb_im = Image.open('./dump/_true_graph.jpg')
#     rgb_arr = pad_im(rgb_im).convert('RGBA')
    return G_true

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

def estimate_graph(masks, nodes, G_gt, per_node=False):
    
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
#                 print((-1, l1))
                G_estimated.add_edges_from([(-1, l1)])

       
    # add missed edges 
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])

    # add edges colors
    colors = []
    mistakes = 0
    per_node_mistakes = defaultdict(int)
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
            mistakes += 1
            per_node_mistakes[k] += 1
            per_node_mistakes[l] += 1
            
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
            mistakes += 1
            per_node_mistakes[k] += 1
            per_node_mistakes[l] += 1
            
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')
    
    per_type_mistakes = defaultdict(int)
    if per_node:
        for k in per_node_mistakes.keys():
            _type = nodes[k]
            per_type_mistakes[_type] += per_node_mistakes[k]
            
        per_node_mistakes
        return mistakes, per_type_mistakes

#     # visualization - debug
#     plt.figure()
#     pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
#     weights = [4 for u, v in G_estimated_complete.edges()]
#     nx.draw(G_estimated_complete, pos, edge_color=colors, edge_labels=edge_labels, linewidths=linewidths, edgecolors=edgecolors, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', width=weights, with_labels=True)
    
#     print('mistakes:', mistakes)
#     plt.tight_layout()
#     plt.savefig('./dump/_fake_graph.jpg', format="jpg")
    
    # rgb_im = Image.open('./dump/_fake_graph.jpg')
    # rgb_arr = pad_im(rgb_im).convert('RGBA')
    # plt.close('all')
    return mistakes

def detailed_viz(masks, nodes, G_gt):
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    for k, label in enumerate(nodes):
        _type = label+1 
        if _type >= 0:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            if _type == 15 or _type == 17:
                node_size.append(500)
            else:
                node_size.append(1000)
    
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
                if iou > 0:
                    doors_rooms_map[k].append((l, iou))    

    # draw connections            
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)

        _conn_top2 = _conn[:2]
        for l, _ in _conn_top2:
            G_estimated.add_edges_from([(k, l)], color='green', weight=4)

        if len(_conn_top2) > 1:
            l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
            G_estimated.add_edges_from([(l1, l2)])
    
    # add missed edges 
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])

    # add edges colors
    colors = []
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')

    # add node-to-node connections
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
    weights = [4 for u, v in G_estimated_complete.edges()]
    nx.draw(G_estimated_complete, pos, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', \
            edge_color=colors, width=weights, with_labels=True)
    plt.tight_layout()
    plt.savefig('./dump/_fake_graph.jpg', format="jpg")
    rgb_im = Image.open('./dump/_fake_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    plt.close('all')
    return rgb_arr, G_estimated_complete


# def draw_masks(masks, real_nodes):
#     bg_img = np.zeros((256, 256, 3)) + 255
#     for m, nd in zip(masks, real_nodes):
#         m[m>0] = 255
#         m[m<0] = 0
#         m = m.detach().cpu().numpy()
#         m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA) 
#         color = ID_COLOR[nd+1]
#         r, g, b = webcolors.name_to_rgb(color)
#         inds = np.array(np.where(m>0))
#         bg_img[inds[0, :], inds[1, :], :] = np.array([[r, g, b]])
#     bg_img = Image.fromarray(bg_img.astype('uint8'))
#     return bg_img


def draw_masks(masks, real_nodes, im_size=256):
    bg_img = np.zeros((256, 256, 3)) + 255
    for m, nd in zip(masks, real_nodes):
        
        # resize map
        m_lg = cv2.resize(m, (256, 256), interpolation = cv2.INTER_AREA) 

        # grab color
        color = ID_COLOR[nd+1]
        r, g, b = webcolors.name_to_rgb(color)

        # draw region
        reg = np.zeros_like(bg_img) + 255
        m_lg = np.repeat(m_lg[:, :, np.newaxis], 3, axis=2)
        m_lg[m_lg>0] = 255
        m_lg[m_lg<0] = 0
        inds = np.where(m_lg > 0)
        reg[inds[0], inds[1], :] = [r, g, b]

        # draw contour
        m_cv = m_lg[:, :, 0].astype('uint8')
        ret,thresh = cv2.threshold(m_cv, 127, 255 , 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:  
            contours = [c for c in contours]
        cv2.drawContours(reg, contours, -1, (0, 0, 0), 1)

        # paste content to background
        inds = np.where(np.prod(reg/255.0, -1) < 1.0)
        bg_img[inds[0], inds[1], :] = reg[inds[0], inds[1], :]
    
    # convert to PIL
    bg_img = Image.fromarray(bg_img.astype('uint8'))
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
generator.load_state_dict(torch.load(checkpoint), strict=False)
generator = generator.eval()

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '../'

# Initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split=phase)
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn, num_workers=1)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Generate state
def gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample, initial_state, true_graph_obj, N):

    # unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = sample

    # configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds

    # set up fixed nodes
    ind_fixed_nodes = torch.tensor(curr_fixed_nodes_state)
    ind_not_fixed_nodes = torch.tensor([k for k in range(real_mks.shape[0]) if k not in ind_fixed_nodes])

    # initialize given masks
    given_masks = torch.zeros_like(real_mks)
    given_masks = given_masks.unsqueeze(1)
    given_masks[ind_not_fixed_nodes.long()] = -1.0
    inds_masks = torch.zeros_like(given_masks)
    given_masks_in = torch.cat([given_masks, inds_masks], 1)    
    real_nodes = np.where(given_nds.detach().cpu()==1)[-1]

    # generate layout
    z = Variable(Tensor(np.random.normal(0, 1, (real_mks.shape[0], opt.latent_dim))))
    best = 9999.0
    best_masks = None
    with torch.no_grad():
        for k in range(N):
            # look for given feats
            if not initial_state:
                print('running state: {}, {}'.format(str(curr_fixed_nodes_state), str(prev_fixed_nodes_state)))
                prev_mks = np.load('{}/feats/feat_opt_vh_{}.npy'.format(PREFIX, '_'.join(map(str, prev_fixed_nodes_state))), allow_pickle=True)
                prev_mks = torch.tensor(prev_mks).cuda().float()   
                given_masks_in[ind_fixed_nodes.long(), 0, :, :] = prev_mks[ind_fixed_nodes.long()]
                given_masks_in[ind_fixed_nodes.long(), 1, :, :] = 1.0
                curr_gen_mks = generator(z, None, given_masks_in, given_nds, given_eds)
            else:
                # print('running initial state')
                curr_gen_mks = generator(z, None, given_masks_in, given_nds, given_eds)

            # save current features
            curr_gen_mks = curr_gen_mks.detach().cpu().numpy()
            mistakes = estimate_graph(curr_gen_mks.copy(), real_nodes, true_graph_obj)
            if mistakes < best:
                best = mistakes
                best_masks = curr_gen_mks.copy()
        np.save('{}/feats/feat_opt_vh_{}.npy'.format(PREFIX, '_'.join(map(str, curr_fixed_nodes_state))), best_masks)
            
    return curr_gen_mks

def run_test(dict_states):
    
    N_states = [1 for k in range(10)]
    dirpath = Path('./FID/test/opt')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.makedirs('./FID/test/opt/', exist_ok=True)
    avg_mistakes = []
    # compute FID for a given sequence
    n_valid_houses = 0
    for i, sample in enumerate(fp_loader):
        if i == 1000:
            break
        mks, nds, eds, _, _ = sample
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
        true_graph_obj = draw_graph([real_nodes, eds.detach().cpu().numpy()])
        
        #### FIX PER ROOM TYPE #####
        # generate final layout initialization
        for j in range(1):
            prev_fixed_nodes_state = []
            curr_fixed_nodes_state = []
            curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample, initial_state=True, true_graph_obj=true_graph_obj, N=1)
            
            # generate per room type
            for _iter in range(10):
                
#                 all_types = [ROOM_CLASS[k]-1 for k in ROOM_CLASS]
#                 sets = {0:'A', 1:'A', 2:'A', 3:'A', 4:'B', 5:'B', 6:'B', 7:'C', 8:'C', 9:'C'}
#                 s = sets[_iter]
#                 _, per_type_mistakes = estimate_graph(curr_gen_mks.copy(), real_nodes, true_graph_obj, per_node=True)
#                 _types = [_t for _t in all_types if (_t not in per_type_mistakes.keys()) and (random.uniform(0, 1) < dict_states['var_v_{}_{}'.format(s, _t)])] # add all without mistakes with some probability
#                 _types += [_t for _t in all_types if (_t in per_type_mistakes.keys()) and (random.uniform(0, 1) < dict_states['var_i_{}_{}'.format(s, _t)])] # add all with mistakes with some probability
                                
                _, per_type_mistakes = estimate_graph(curr_gen_mks.copy(), real_nodes, true_graph_obj, per_node=True)
                _types = [_t for _t in all_types if (_t not in per_type_mistakes.keys()) and (_iter >= dict_states['var_v_{}'.format(_t)])] # add valid after k-th iteration
                _types += [_t for _t in all_types if (_t in per_type_mistakes.keys()) and (_iter >= dict_states['var_i_{}'.format(_t)])] # add invalid after k-th iteration
                
                if len(_types) > 0:
                    curr_fixed_nodes_state = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types])
                else:
                    curr_fixed_nodes_state = np.array([])
                curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample, initial_state=False, true_graph_obj=true_graph_obj, N=1)
                prev_fixed_nodes_state = list(curr_fixed_nodes_state)

            mistakes = estimate_graph(curr_gen_mks.copy(), real_nodes, true_graph_obj)
            avg_mistakes.append(mistakes)
            if mistakes == 0:
                n_valid_houses += 1
            imk = draw_masks(curr_gen_mks.copy(), real_nodes)
            imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0
            save_image(imk, './FID/test/opt/{}_{}.png'.format(i, j), nrow=1, normalize=False)
            
            
    fid_value = calculate_fid_given_paths(['./FID/val/', './FID/test/opt/'], 2, 'cpu', 2048)
    ratio_valid_houses = 1.0-n_valid_houses/1000.0
#     avg_mistakes = np.mean(fid_value)
    out_str = "all_states {} {}\n".format(str(dict_states), fid_value)
    print(out_str)
    with open('./FID/opt_results_fid_dynamic.txt', 'a') as f:
        f.write("{}\n".format(out_str))
    return fid_value

# Bayesian optimization
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
all_types = [ROOM_CLASS[k]-1 for k in ROOM_CLASS]
space = {}

# # minimize the objective over the space
# for s in ['A', 'B', 'C']:
#     for _t in all_types:
#         space['var_v_{}_{}'.format(s, _t)] = hp.uniform('var_v_{}_{}'.format(s, _t), 0.0, 1.0)   
#         space['var_i_{}_{}'.format(s, _t)] = hp.uniform('var_i_{}_{}'.format(s, _t), 0.0, 1.0)   
        
        
# minimize the objective over the space
for _t in all_types:
    space['var_v_{}'.format(_t)] = hp.choice('var_v_{}'.format(_t), tuple(range(10)))
    space['var_i_{}'.format(_t)] = hp.choice('var_i_{}'.format(_t), tuple(range(10)))
            


# minimize the objective over the space
best = fmin(run_test, space, algo=tpe.suggest, max_evals=1000)
print(best)