import argparse
import os
import numpy as np
import math
import sys
import random
import torch
from reconstruction_dataset import AutoencoderDataset, Autoencoder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from models import Discriminator, Generator, compute_gradient_penalty, weights_init_normal
import matplotlib.pyplot as plt




# Initialize dataset iterator
checkpoint = '/home/nelson/Workspace/autodesk/housegan2/checkpoints/gen_housegan_E_1000000.pth'
ae_dataset = AutoencoderDataset(transform=transforms.Normalize(mean=[0.5], std=[0.5]))
batch_size = 1
ae_loader = torch.utils.data.DataLoader(ae_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)
generator = Generator().cuda()
update_freq = 256
pretrained_dict = torch.load(checkpoint)
model_dict = generator.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
generator.load_state_dict(model_dict)


optimizer = optim.Adam(generator.parameters(), lr=0.0001)
optimizer.zero_grad()
running_loss = 0.0
for k in range(1000):
    for i, batch in enumerate(ae_loader):

        # retrieve data
        prev_feats, mask, nodes, edges = batch
        prev_feats, mask = Variable(prev_feats).squeeze(0).cuda(), Variable(mask).squeeze(0).cuda()
        given_nds, given_eds = Variable(nodes).squeeze(0).cuda(), Variable(edges).squeeze(0).cuda()

        # generate random state
        N = np.random.randint(given_nds.shape[1], size=1)
        fixed_nodes_state = torch.tensor(np.random.choice(list(range(given_nds.shape[1])), size=N, replace=False)).cuda()
        # print('running state: {}'.format(str(fixed_nodes_state)))
        z = Variable(torch.Tensor(np.random.normal(0, 1, (mask.shape[0], 128)))).cuda()
        gen_mks, curr_feats = generator(z, given_nds, given_eds, given_v=prev_feats, state=fixed_nodes_state)
        
        # reconstruction loss
        target_mask = mask[fixed_nodes_state]
        recon_mask = gen_mks[fixed_nodes_state]

        if (k % 10 == 0):
            # debug masks
            for m_t, m_r, f in zip(target_mask, recon_mask, range(len(fixed_nodes_state))):
                if i > 10:
                    continue
                m_t = m_t.detach().cpu().numpy()*255.0
                m_r = m_r.detach().cpu().numpy()*255.0
                fig = plt.figure(figsize=(12, 6))
                fig.add_subplot(1, 2, 1)
                plt.imshow(np.rot90(m_r, 2))
                fig.add_subplot(1, 2, 2)
                plt.imshow(np.rot90(m_t, 2))
                plt.savefig('./debug/debug_{}_{}.png'.format(k*len(ae_dataset)+i, f))
                plt.close('all')

        loss = nn.MSELoss(reduction='none')
        out = loss(recon_mask.view(batch_size, -1), target_mask.view(batch_size, -1)).sum()
        out.backward()
        running_loss += out

        step = k*len(ae_dataset)+i
        if (step + 1)%update_freq == 0:
            print(step)
            optimizer.step()
            optimizer.zero_grad()
            print('[%d] loss: %.3f' %
                (k + 1, running_loss / len(ae_dataset)))
            running_loss = 0

    if (k + 1) % 50 == 0:
        torch.save(generator.state_dict(), './finetuned_generator_{}.pth'.format(k))




