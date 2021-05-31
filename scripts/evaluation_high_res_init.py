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
from models.models_exp_high_res import Generator
# from models_exp_3 import Generator

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


ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}


target_set = 8
phase='eval'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/gen_housegan_E_1000000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/functional_graph_fixed_A_300000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_functional_graph_with_l1_loss_attempt_3_A_550000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_128_A_750000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_with_doors_64x64_per_room_type_A_230000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_with_doors_64x64_per_room_type_attempt_2_A_360000.pth'

checkpoint = '../exp_random_types_attempt_3_A_500000_G.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_high_res_with_doors_64x64_per_room_type_attempt_2_A_360000.pth'


# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_per_room_type_enc_dec_plus_local_A_260000.pth'
# checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/exp_autoencoder_A_72900.pth'

PREFIX = "/home/nelson/Workspace/autodesk/automated_floorplan/"
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
        elif m > 0 and (_type_k==15 or _type_l==15) and (_type_l!=17 and _type_k!=17):
            if _type_k==15:
                G_true.add_edges_from([(l, -1)])   
            elif _type_l==15:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    
    # # visualization - debug
    print(len(node_size))
    print(len(colors_H))
    print(len(linewidths))
    print(G_true.nodes())
    print(g_true[0])
    print(len(edgecolors))
    

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
                print((-1, l1))
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
    
    print('mistakes:', mistakes)
    plt.tight_layout()
    plt.savefig('./dump/_fake_graph.jpg', format="jpg")
    
    rgb_im = Image.open('./dump/_fake_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    plt.close('all')

    return mistakes, rgb_im


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
                                        shuffle=False, collate_fn=floorplan_collate_fn)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Generate state
def gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample):

    # unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = sample

    # configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds
    print(given_nds)
    asdasd
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
    with torch.no_grad():

        # look for given feats
        if len(curr_fixed_nodes_state) > 0:
            print('running state: {}, {}'.format(str(curr_fixed_nodes_state), str(prev_fixed_nodes_state)))
            prev_mks = np.load('{}/feats/feat_{}.npy'.format(PREFIX, '_'.join(map(str, prev_fixed_nodes_state))), allow_pickle=True)
            prev_mks = torch.tensor(prev_mks).cuda().float()   
            given_masks_in[ind_fixed_nodes.long(), 0, :, :] = prev_mks[ind_fixed_nodes.long()]
            given_masks_in[ind_fixed_nodes.long(), 1, :, :] = 1.0
            asdasd
            curr_gen_mks = generator(z, given_masks_in, given_nds, given_eds)
        else:
            print('running initial state')
            curr_gen_mks = generator(z, given_masks_in, given_nds, given_eds)

        # reconstruct
        curr_gen_mks = curr_gen_mks.detach().cpu().numpy()
        print(curr_gen_mks.shape)
        fake_im = draw_masks(curr_gen_mks.copy(), real_nodes)
        
        # save current features
        np.save('{}/feats/feat_{}.npy'.format(PREFIX, '_'.join(map(str, curr_fixed_nodes_state))), curr_gen_mks)
        
    return fake_im, curr_gen_mks

#  Vectorize
globalIndex = 0
final_images = []
page_count = 0
n_rows = 0
feats_tensor = []
all_images = []
all_dists = []
for i, sample in enumerate(fp_loader):
    if i != 15:
        continue

    # draw real graph and groundtruth
    mks, nds, eds, _, _ = sample
    np.save('./debug.npy', {'nds': nds, 'eds': eds})

    real_nodes = np.where(nds.detach().cpu()==1)[-1]
    graph = [real_nodes, None]
    true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
    graph_im.save('{}/figure_2/graph_{}.png'.format(PREFIX, i))
    # real_im = draw_masks(mks.detach().cpu().numpy(), real_nodes)
    # real_im.save('{}/runs/gt.png'.format(PREFIX))

    # CUSTOM SEQUENCE
    all_states = [[9, 15], [9, 15], [9, 15], [3, 9, 15], [1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 9, 15], [0, 1, 2, 3, 5, 9, 15], [0, 1, 2, 3, 5, 6, 9, 15], [0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16]] # best for exp_random_types_attempt_3_A_500000_G - FID

    # # pick a sequence
    # print('** RANDOM **')
    # all_types = [ROOM_CLASS[k]-1 for k in ROOM_CLASS]
    # all_states = [[t for t in all_types if random.uniform(0, 1) > 0.5] for _ in range(10)] # RANDOM SCHEME

    #### FIX PER ROOM TYPE #####
    # generate final layout initialization
    os.makedirs('./figure_2/', exist_ok=True)
    start = time.time()
    end = time.time()
    _round = 0
    while True and (end - start < 30000) and _round < 20:
        print(end - start)
        prev_fixed_nodes_state = []
        curr_fixed_nodes_state = []
        im0, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0
        # save_image(im0, './figure_2/fp_init.png'.format(), nrow=1, normalize=False)

        # generate per room type
        for _iter, _types in enumerate(all_states):
            if len(_types) > 0:
                curr_fixed_nodes_state = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types])
            else:
                curr_fixed_nodes_state = np.array([])
            imk, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
            imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0

            print(real_nodes)
            print(curr_fixed_nodes_state, _types)
            asdasd
            prev_fixed_nodes_state = list(curr_fixed_nodes_state)
            
            # if _iter == 3 or _iter == 6:
            #     save_image(imk, './figure_2/fp_{}_inter.png'.format(_iter), nrow=1, normalize=False)

        # save final floorplans
        miss, gim = estimate_graph(curr_gen_mks, real_nodes, true_graph_obj)
        if miss >= 0:
            
            # imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0
            gim = torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0

            np.save('./figure_2/masks_{}.npy'.format(_round), {'nodes':nds, 'edges': eds, 'masks':curr_gen_mks})
            save_image(imk, './figure_2/fp_graph_{}_iter_{}_final.png'.format(_iter, _round), nrow=1, normalize=False)
            save_image(gim, './figure_2/gp_graph_{}_iter_{}_final.png'.format(_iter, _round), nrow=1, normalize=False)
            _round += 1
        
            asdasd
        end = time.time()
    asdasd

    ####  FIX ALL NODES ####
    # generate final layout initialization
    # for _ in range(20):
    #     prev_fixed_nodes_state = []
    #     curr_fixed_nodes_state = []
    #     im0, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
    #     gim, _ = detailed_viz(curr_gen_mks, real_nodes, true_graph_obj)
    #     all_images.append(torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0)
    #     all_images.append(torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0)

    #     # generate per room type
    #     room_types = list(set(real_nodes))
    #     for _ in range(len(room_types)):
            
    #         curr_fixed_nodes_state = all_nodes
    #         imk, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
    #         gim, _ = detailed_viz(curr_gen_mks, real_nodes, true_graph_obj)
    #         prev_fixed_nodes_state = list(curr_fixed_nodes_state)
    #     all_images.append(torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0)
    #     all_images.append(torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0)

    #     prev_fixed_nodes_state = all_nodes
    #     curr_fixed_nodes_state = all_nodes
    #     imk, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
    #     gim, _ = detailed_viz(curr_gen_mks, real_nodes, true_graph_obj)
    #     all_images.append(torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0)
    #     all_images.append(torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0)


    # ### RANDOM SELECTION OF NODES ####

    # # sample N different sequences
    # random_state = list(all_nodes)
    # random.shuffle(random_state)
    # prev_fixed_nodes_state = []
    # curr_fixed_nodes_state = []
    # im0, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
    # gim, _ = detailed_viz(curr_gen_mks, real_nodes, true_graph_obj)
    # all_images.append(torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0)
    # all_images.append(torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0)

    # for s in random_state:
    #     prev_fixed_nodes_state = list(curr_fixed_nodes_state)
    #     curr_fixed_nodes_state.append(s)
    #     imk, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
    #     gim, _ = detailed_viz(curr_gen_mks, real_nodes, true_graph_obj)
    #     all_images.append(torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0)
    #     all_images.append(torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0)

    # prev_fixed_nodes_state = all_nodes
    # curr_fixed_nodes_state = all_nodes
    # imk, curr_gen_mks = gen_state(curr_fixed_nodes_state, prev_fixed_nodes_state, sample)
    # gim, _ = detailed_viz(curr_gen_mks, real_nodes, true_graph_obj)
    # all_images.append(torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0)
    # all_images.append(torch.tensor(np.array(gim).transpose((2, 0, 1)))/255.0)

# all_images = torch.stack(all_images)
# save_image(all_images, '{}/runs/exp_add_doors_random_64x64_g2.png'.format(PREFIX), nrow=15, normalize=False)
