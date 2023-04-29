from numpy import ceil
from typing import List, Union

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

from .buildingblocks import ConvBlock, ResNetBlock

class Generator(nn.Module):
    """ The CNN used as the generator in the model.
    TODO -- add a better description.
    """
    def __init__(self, input_nc: int, output_nc: int,
                 n_filters: int = 32) -> None:
        """ Constructs a new generator network.

        Parameters:
            input_nc (int) -- number of channels of the input tensor
            out
            n_filters (int) -- number of filters of the first convolutional layer. Default: 32
        """
        super(Generator, self).__init__()
        self.input_block = nn.Sequential(nn.ReflectionPad2d(3),
                                         ConvBlock(input_nc, n_filters,
                                                   norm_type='instance',
                                                   conv_kernel_size=7,
                                                   conv_stride=1,
                                                   conv_padding=0))

        self.down_conv_blocks = nn.Sequential(ConvBlock(n_filters, 2 * n_filters,
                                                        norm_type='instance'),
                                              ConvBlock(2 * n_filters, 4 * n_filters,
                                                        norm_type='instance'),
                                              ConvBlock(4 * n_filters, 8 * n_filters,
                                                        norm_type='instance'))

        self.resnet_blocks = nn.Sequential(*([ResNetBlock(8 * n_filters)] * 9))

        self.up_conv_blocks = nn.Sequential(ConvBlock(8 * n_filters, 4 * n_filters,
                                                      conv_type='up_conv',
                                                      norm_type='instance'),
                                            ConvBlock(8 * n_filters, 2 * n_filters,
                                                      conv_type='up_conv',
                                                      norm_type='instance'),
                                            ConvBlock(4 * n_filters, n_filters,
                                                      conv_type='up_conv',
                                                      norm_type='instance'))

        self.output_block = nn.Sequential(nn.ReflectionPad2d(3),
                                          ConvBlock(n_filters, input_nc,
                                                    activation_type='none',
                                                    conv_kernel_size=7,
                                                    conv_stride=1,
                                                    conv_padding=0),
                                          nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """ The default forward
        TODO -- add a better description

        Parameters:
            x (Tensor) -- the input tensor that represents a image with droplets
        Retunr (Tensor) -- the new generated image
        """
        y = self.input_block(x)
        down_conv_outputs = []

        for i in range(len(self.down_conv_blocks)):
            y = self.down_conv_blocks[i](y)
            down_conv_outputs += [y]

        y = self.resnet_blocks(y)

        y = self.up_conv_blocks[0](y)
        y = torch.cat((y, down_conv_outputs[1]), dim=1)
        y = self.up_conv_blocks[1](y)
        y = torch.cat((y, down_conv_outputs[0]), dim=1)
        y = self.up_conv_blocks[2](y)
    
        y = self.output_block(y)

        return y


class Discriminator(nn.Module):
    """ The CNN used as discriminator in the model.

    It is a PatchGAN with a 70x70 effective receptive field"""
    def __init__(self, input_nc: int, n_filters: int = 32) -> None:
        """ Constructs a new discriminator network.

        Parameters:
            input_nc (int) -- number of channels of the input tensor
            n_filters (int) -- number of filters of the first convolutional layer. Default: 32
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(ConvBlock(input_nc, n_filters, 
                                             conv_kernel_size=4,
                                             conv_padding=2,
                                             activation_type='lrelu'),
                                   ConvBlock(n_filters, 2 * n_filters,
                                             conv_kernel_size=4,
                                             conv_padding=2,
                                             norm_type='instance',
                                             activation_type='lrelu'),
                                   ConvBlock(2 * n_filters, 4 * n_filters,
                                             conv_kernel_size=4, 
                                             conv_padding=2,
                                             norm_type='instance',
                                             activation_type='lrelu'),
                                   ConvBlock(4 * n_filters, 8 * n_filters,
                                             conv_kernel_size=4, 
                                             conv_stride=1,
                                             conv_padding=2,
                                             norm_type='instance',
                                             activation_type='lrelu'),
                                   ConvBlock(8 * n_filters, 1, 
                                             conv_kernel_size=4,
                                             conv_stride=1,
                                             conv_padding=2,
                                             activation_type='lrelu'),
                                    nn.Sigmoid())

    def forward(self, x: Tensor,
                extract_feat: bool = False) -> Union[Tensor, List[Tensor]]:
        """ Passes the input through the discriminator and returns a list
        containing the output of each layer

        Parameters:
            x (Tensor) -- the input tensor
            extract_feat (bool) -- whether to extrect or not intermediate features.
                                   Default: False.
        Return (Tensor or List[Tensor]) -- the output of the discriminator or
                                           the output each layer of the discriminator
        """
        if extract_feat:
            return_layer = {str(i) : 'layer' + str(i) \
                            for i in range(len(self.model))}
            feat_extractor = IntermediateLayerGetter(self.model, return_layer)
            y = [feat for feat in feat_extractor(x).values()]
            return y
        return self.model(x)


class MultScaleDiscriminator(nn.Module):
    """ TODO """
    def __init__(self, input_nc: int = 6) -> None:
        super(MultScaleDiscriminator, self).__init__()
        """
        TODO
        """
        self.conv_blocks = nn.Sequential(ConvBlock(input_nc, 64,
                                                   norm_type='batch',
                                                   activation_type='prelu',
                                                   pooling_type='max'),
                                         ConvBlock(64, 256,
                                                   norm_type='batch',
                                                   activation_type='prelu',
                                                   pooling_type='max'),
                                         ConvBlock(256, 512,
                                                   norm_type='batch',
                                                   activation_type='prelu',
                                                   pooling_type='max'),
                                         ConvBlock(512, 64,
                                                   norm_type='batch',
                                                   activation_type='prelu',
                                                   pooling_type='max'))

        self.downsample_block = nn.MaxPool2d(4, stride=2, padding=1)

        self.upsample_blocks = nn.ModuleList([nn.Upsample(scale_factor=2**i)
                                              for i in range(1, 5)])

        self.output_block = nn.Sequential(nn.Conv2d(64, 72, kernel_size=1),
                                          nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """ TODO """
        y = self.conv_blocks(x)

        y_downsampled = y
        for upsample_block in self.upsample_blocks:
            y_downsampled = self.downsample_block(y_downsampled)
            y_upsampled = upsample_block(y_downsampled)
            y = torch.cat((y, y_upsampled), dim=1)

        y = self.output_block(y)
        return y
