from torch import nn
from torch.nn.utils import spectral_norm
import torch

# from torch.legacy.nn import Identity

# Residual network.
# WGAN-GP paper defines a residual block with up & downsampling.
# See the official implementation (given in the paper).
# I use architectures described in the official implementation,
# since I find it hard to deduce the blocks given here from the text alone.

def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=True):
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
    return block

class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
        out = self.model(out)
        return out

class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = self.model(x)
        out = (out[:,:,::2,::2] + out[:,:,1::2,::2] + out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
        return out

class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        )
    def forward(self, x):
        x = x.repeat((1, 4, 1, 1)) # Weird concat of WGAN-GPs upsampling process.
        out = self.model(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim] # Define the dimensions for layer normalization.
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = None # Identity
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

        self.model = nn.Sequential(
            nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)

class DiscBlock1(nn.Module):
    def __init__(self, n_output):
        super(DiscBlock1, self).__init__()

        self.conv1 = nn.Conv2d(3, n_output, 3, padding=(3-1)//2)
        self.conv2 = ConvMeanPool(n_output, n_output, 1)
        self.conv_shortcut = MeanPoolConv(3, n_output, 1)

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2
        )

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(                     # 128 x 1 x 1
            nn.ConvTranspose2d(128, 128, 4, 1, 0),      # 128 x 4 x 4
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 8 x 8
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 16 x 16
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 32 x 32
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=(3-1)//2),     # 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img

class GeneratorMod(nn.Module):
    def __init__(self):
        super(GeneratorMod, self).__init__()

        self.dec_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 1, 0),      # 128 x 4 x 4
            ResidualBlock(128, 16, 3, resample='up'),  # 128 x 8 x 8
            )

        self.model = nn.Sequential(                    
            ResidualBlock(32, 128, 3, resample='up'),  # 128 x 16 x 16
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 32 x 32
            ResidualBlock(128, 128, 3, resample='up'),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=(3-1)//2),     # 3 x 64 x 64
            nn.Tanh()
        )

        self.enc_1 = nn.Sequential(
            *conv_block(4, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 16, 3, 2, 1, act="leaky")) 

    def forward(self, z, y):
        z = self.dec_1(z)
        y = self.enc_1(y)
        x = torch.cat([z, y], 1)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        n_output = 128
        '''
        This is a parameter but since we experiment with a single size
        of 3 x 32 x 32 images, it is hardcoded here.
        '''
        
        self.DiscBlock1 = DiscBlock1(n_output)                      # 128 x 16 x 16

        self.model = nn.Sequential(
            ResidualBlock(n_output, n_output, 3, resample='down', bn=False, spatial_dim=32),  # 128 x 8 x 8
            ResidualBlock(n_output, n_output, 3, resample='down', bn=False, spatial_dim=16),  # 128 x 8 x 8
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),    # 128 x 8 x 8
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),    # 128 x 8 x 8
            nn.ReLU(inplace=True),
        )
        self.l1 = nn.Sequential(nn.Linear(128, 1))                  # 128 x 1

    def forward(self, x):
        # x = x.view(-1, 3, 32, 32)
        y = self.DiscBlock1(x)
        y = self.model(y)
        y = y.view(x.size(0), 128, -1)
        y = y.mean(dim=2)
        out = self.l1(y)#.unsqueeze_(1).unsqueeze_(2) # or *.view(x.size(0), 128, 1, 1, 1)
        return out
