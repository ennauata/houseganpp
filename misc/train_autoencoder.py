import argparse
import os
import numpy as np
import math
import sys
import random
import torch
from autoencoder_dataset import AutoencoderDataset, Autoencoder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

# Initialize dataset iterator
model = Autoencoder().cuda()
ae_dataset = AutoencoderDataset(transform=transforms.Normalize(mean=[0.5], std=[0.5]))
batch_size = 1024
ae_loader = torch.utils.data.DataLoader(ae_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001)


for k in range(1000):
    running_loss = 0.0
    for i, batch in enumerate(ae_loader):
        y, x = batch
        y, x = Variable(y), Variable(x)
        y_hat = model(x)
        # print(y_hat[:3, :3, :3, :3])
        # print(y[:3, :3, :3, :3])
        loss = nn.L1Loss(reduction='none')
        out = loss(y_hat.view(batch_size, -1), y.view(batch_size, -1)).sum()
        out.backward()
        optimizer.step()
        running_loss += out.item()

    print('[%d] loss: %.3f' %
        (k + 1, running_loss / len(ae_dataset)))

