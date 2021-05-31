import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps
import torch.nn.utils.spectral_norm as spectral_norm
from models.model_resnet import ResidualBlock

def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, *x.shape[1:]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1, 1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, x, x_fake, given_y=None, given_w=None, \
                             nd_to_sample=None, data_parallel=None, \
                             ed_to_sample=None, ddim=2):

    indices = nd_to_sample, ed_to_sample
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    u = torch.FloatTensor(*([x.shape[0]]+[1]*ddim)).to(device)
    u.data.resize_(*([x.shape[0]]+[1]*ddim))
    u.uniform_(0, 1)
    x_both = x.data*u + x_fake.data*(1-u)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    if data_parallel:
        _output = data_parallel(D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        _output = D(x_both, given_y, given_w, nd_to_sample)
    grad = torch.autograd.grad(outputs=_output, inputs=x_both, grad_outputs=grad_outputs, \
                               retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, batch_norm=True, ddim='2D'):
    block = []

    if ddim=='2D':
        conv = torch.nn.Conv2d
        tconv = torch.nn.ConvTranspose2d
    elif ddim=='3D':
        conv = torch.nn.Conv3d
        tconv = torch.nn.ConvTranspose3d

    if upsample:
        if spec_norm:
            block.append(spectral_norm(tconv(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True)))
        else:
            block.append(tconv(in_channels, out_channels, \
                                                   kernel_size=k, stride=s, \
                                                   padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(conv(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True)))
        else:        
            block.append(conv(in_channels, out_channels, \
                                                       kernel_size=k, stride=s, \
                                                       padding=p, bias=True))
    if batch_norm:
        block.append(nn.InstanceNorm2d(out_channels))
        # block.append(nn.BatchNorm2d(out_channels))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(inplace=True))
    elif "tanh":
        block.append(torch.nn.Tanh())
    return block

class CMP2D(nn.Module):
    def __init__(self, in_channels):
        super(CMP2D, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            ResidualBlock(3*in_channels, 3*in_channels, 3, resample=None), 
            ResidualBlock(3*in_channels, 3*in_channels, 3, resample=None), 
            *conv_block(3*in_channels, in_channels, 3, 1, 1, act="relu", ddim='2D'))

    def forward(self, feats, edges=None):
        
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        
        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = pooled_v_pos.scatter_add(0, pos_v_dst, pos_vecs_src)
        
        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = pooled_v_neg.scatter_add(0, neg_v_dst, neg_vecs_src)
        
        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out
    

class CMP3D(nn.Module):
    def __init__(self, in_channels):
        super(CMP3D, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky", ddim='3D'),
            *conv_block(2*in_channels, 2*in_channels, 3, 1, 1, act="leaky", ddim='3D'),
            *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky", ddim='3D'))
             
    def forward(self, feats, edges=None):
        
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)

        pooled_v_pos = torch.zeros(V, feats.shape[-4], feats.shape[-1], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, feats.shape[-4], feats.shape[-1], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        
        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = pooled_v_pos.scatter_add(0, pos_v_dst, pos_vecs_src)

        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = pooled_v_neg.scatter_add(0, neg_v_dst, neg_vecs_src)
        
        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)

        return out
    
class Generator(nn.Module):
    def __init__(self, ddim='2D', hidden_dim=16):
        super(Generator, self).__init__()
        if ddim == '2D':
            self.dim = 2
        elif ddim == '3D':
            self.dim = 3

        self.init_size = 8
        self.hidden_dim = hidden_dim
        self.l1 = nn.Sequential(nn.Linear(146, hidden_dim//2 * self.init_size ** self.dim))
        self.upsample_1 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')
        self.upsample_2 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')
        self.upsample_3 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')
        self.upsample_4 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')

        if ddim=='2D':
            self.cmp_1 = CMP2D(in_channels=hidden_dim)
            self.cmp_2 = CMP2D(in_channels=hidden_dim)
            self.cmp_3 = CMP2D(in_channels=hidden_dim)
            self.cmp_4 = CMP2D(in_channels=hidden_dim)
            self.cmp_5 = CMP2D(in_channels=hidden_dim)

        elif ddim=='3D':
            self.cmp_1 = CMP3D(in_channels=hidden_dim)
            self.cmp_2 = CMP3D(in_channels=hidden_dim)
            self.cmp_3 = CMP3D(in_channels=hidden_dim)

        self.encoder = nn.Sequential(
            *conv_block(2, hidden_dim//2, 3, 2, 1, act="relu", ddim=ddim),
            ResidualBlock(hidden_dim//2, hidden_dim//2, 3, resample='down'),
            # ResidualBlock(hidden_dim//2, hidden_dim//2, 3, resample='down'),
            ResidualBlock(hidden_dim//2, hidden_dim//2, 3, resample='down')) 

        self.head = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            *conv_block(hidden_dim, 1, 3, 1, 1, act="tanh", ddim=ddim, batch_norm=False))                                        

    def forward(self, z, given_p=None, given_y=None, given_w=None, given_v=None, state=None):

        # reshape noise
        z = z.view(-1, 128)

        # include nodes
        if True:
            y = given_y.view(-1, 18) #10
            z = torch.cat([z, y], 1)
        x = self.l1(z)              
        f = x.view(*([-1, self.hidden_dim//2] + [self.init_size]*self.dim))

        # combine masks and noise vectors
        p = self.encoder(given_p)
        f = torch.cat([f, p], 1)

        # Conv-MPN: message-passing
        x = self.cmp_1(f, given_w).view(-1, *f.shape[1:])
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])   
        x = self.upsample_2(x)
        x = self.cmp_3(x, given_w).view(-1, *x.shape[1:])   
        x = self.upsample_3(x)
        x = self.cmp_4(x, given_w).view(-1, *x.shape[1:])  

        # generation head
        x = self.head(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, *x.shape[2:])

        return x

class Discriminator(nn.Module):
    def __init__(self, ddim='2D', hidden_dim=16, in_dim=64):
        super(Discriminator, self).__init__()

        if ddim == '2D':
            self.dim = 2
        elif ddim == '3D':
            self.dim = 3

        if ddim == '2D':
            self.cmp_1 = CMP2D(in_channels=hidden_dim)
            self.cmp_2 = CMP2D(in_channels=hidden_dim)
            self.cmp_3 = CMP2D(in_channels=hidden_dim)
            self.cmp_4 = CMP2D(in_channels=hidden_dim)
            self.cmp_5 = CMP2D(in_channels=hidden_dim)

        elif ddim == '3D':
            self.cmp_1 = CMP3D(in_channels=hidden_dim)
            self.cmp_2 = CMP3D(in_channels=hidden_dim)
            self.cmp_3 = CMP3D(in_channels=hidden_dim)  

        self.l1 = nn.Sequential(nn.Linear(18, 8 * in_dim ** self.dim))
        self.downsample_1 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_2 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_3 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_4 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_5 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')

        self.encoder = nn.Sequential(
            *conv_block(9, hidden_dim, 3, 1, 1, act="relu"),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None))
        
        # define classification heads
        self.head_local_cnn = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'))
        self.head_local_l1 = nn.Sequential(nn.Linear(hidden_dim, 1))

        self.head_global_cnn = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'))
        self.head_global_l1 = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
        S = x.shape[-1]
        x = x.view(*([-1, 1] + [S]*self.dim))

        # include nodes
        if True:
            y = given_y
            y = self.l1(y)
            y = y.view(*([-1, 8] + [S]*self.dim))
            x = torch.cat([x, y], 1)

        # Conv-MPN: message-passing
        x = self.encoder(x)
        x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])  
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_2(x)
        x = self.cmp_3(x, given_w).view(-1, *x.shape[1:])
        x = self.downsample_4(x)
        x_l = self.cmp_4(x, given_w).view(-1, *x.shape[1:])

        # global classification head
        x_g = add_pool(x_l, nd_to_sample)
        x_g = self.head_global_cnn(x_g)
        validity_global = self.head_global_l1(x_g.view(-1, x_g.shape[1]))

        # local classification head
        if True:
            x_l = self.head_local_cnn(x_l)
            x_l = add_pool(x_l, nd_to_sample)
            validity_local = self.head_local_l1(x_l.view(-1, x_l.shape[1]))
            validity = validity_global + validity_local
            return validity
        else:
            return validity_global
    
