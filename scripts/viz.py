import argparse
import os
import numpy as np
import math
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import svgwrite
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob
import cv2
import webcolors
import time


ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}


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
    
    # # # visualization - debug
    # print(len(node_size))
    # print(len(colors_H))
    # print(len(linewidths))
    # print(G_true.nodes())
    # print(g_true[0])
    # print(len(edgecolors))
    

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