""" This module contains the implementations of all the basic building
blocks used in the implementation of the de-raining model.
"""
import torch
from torch import Tensor
import torch.nn as nn

from typing import Dict, Union, List, Tuple

class ConvBlock(nn.Module):
    ''' Groups together a convolutional, a normalization and optional
    activation and pooling operations to form the basic convolutional layer used in
    all the networks.
    '''
    def __init__(self, input_nc: int, output_nc: int,
                 conv_type: str = 'down_conv',
                 norm_type: str = 'none',
                 activation_type: str = 'lrelu',
                 pooling_type: str = 'none',
                 conv_kernel_size: Union[int, Tuple[int, int]] = 3,
                 conv_stride: Union[int, tuple] = 2,
                 padding_mode: str = 'zeros',
                 conv_padding: Union[int, tuple, str] = 1,
                 output_padding: Union[int, tuple, str] = 1,
                 pool_kernel_size: Union[int, Tuple[int, int]] = 3,
                 pool_stride: Union[int, Tuple[int, int]] = 3,
                 pool_padding: Union[int, tuple, str] = 0) -> None:
        """ The constructor of the convolutional block.

        Parameters:
            input_nc(int) -- number of input channels for the block
            output_nc(int) -- number of output channels for the block
            conv_type(str) -- whether the block is a up (ConvTranspose2d) or downsampling (Conv2d) convolutional
                              layer. Could be 'up_conv' or 'down_conv'. Default: 'down_conv'
            norm_type (str) -- the type of normalization to be used in the block.
            Could be 'instance', 'batch' or 'none'. Default: 'none'
            activation_type(str) -- which type of activation to use in the block.
                                    Could be 'relu', 'lrelu', 'prelu' or 'none'.
                                    Default: 'none'.
            pooling_type (str) -- which type of pooling to use in the block.
                                  Could be 'max' or 'none'. Default: 'none'.
            conv_kernel_size(int or tuple) -- the size of the kernel used in the convolution
                                              operation. Default: 3
            conv_stride(int or tuple, optional) -- the stride used in the convolution operation. Default: 2
            padding_mode (str) -- 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
            conv_padding(int, tuple or str, optional) -- padding added to the input of the convolution operation. Default: 0
            output_padding(int, tuple or str, optional) -- padding added to the output of a upsampling convolution. Default: 0
            pool_kernel_size(int or tuple) -- the size of the kernel used in the pooling
                                              operation. Default: 3
            pool_stride(int or tuple, optional) -- the stride used in the pooling operation. Default: 3
            pool_padding(int, tuple or str, optional) -- padding added to the input of the pooling operation. Default: 1 
        """
        super(ConvBlock, self).__init__()

        conv_block = []
        if conv_type == 'up_conv':
            conv_block += [nn.ConvTranspose2d(input_nc, output_nc,
                                              kernel_size=conv_kernel_size,
                                              stride=conv_stride,
                                              padding_mode=padding_mode,
                                              padding=conv_padding,
                                              output_padding=output_padding)]
        elif conv_type == 'down_conv':
            conv_block += [nn.Conv2d(input_nc, output_nc,
                                     kernel_size=conv_kernel_size,
                                     stride=conv_stride,
                                     padding_mode=padding_mode,
                                     padding=conv_padding)]
        else:
            raise ValueError("conv_type sould be 'up_conv' or 'down_conv'")

        if norm_type == 'instance':
            conv_block += [nn.InstanceNorm2d(output_nc)]
        elif norm_type == 'batch':
            conv_block += [nn.BatchNorm2d(output_nc)]
        elif norm_type != 'none':
            raise ValueError("norm_type should be 'instance', 'batch' or 'none'")

        if activation_type == 'relu':
            conv_block += [nn.ReLU(inplace=True)]
        elif activation_type == 'lrelu':
            conv_block += [nn.LeakyReLU(negative_slope=0.2,
                                        inplace=True)]
        elif activation_type == 'prelu':
            conv_block += [nn.PReLU()]
        elif activation_type != 'none':
            raise NotImplementedError('This type of activation is not implemented yet')

        if pooling_type == 'max':
            conv_block += [nn.MaxPool2d(pool_kernel_size, 
                                        stride=pool_stride, 
                                        padding=pool_padding)]
        elif pooling_type != 'none':
            raise NotImplementedError('This type of pooling is not implemented yet')
        
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: Tensor) -> Tensor:
        """ Passes the input through all the layers of the block

        Parameters:
            x -- the input tensor
        Return (Tensor) -- output tensor
        """
        return self.conv_block(x)

class ResNetBlock(nn.Module):
    """ The residual network block used in the models.

    TODO -- Describe why ResNet blocks is needed.
    """
    def __init__(self, input_nc: int) -> None:
        """ Initializes a new ResNet block.

        Parameters:
            input_nc (int) -- number of input channels
        """
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(ConvBlock(input_nc, input_nc, 
                                             norm_type='instance',
                                             conv_stride=1),
                                   ConvBlock(input_nc, input_nc, 
                                             norm_type='instance',
                                             activation_type='none',
                                             conv_stride=1))
    def forward(self, x: Tensor) -> Tensor:
        """ Passes the input through the ResNet block.

        Parameters:
            x (Tensor) -- the input tensor
        Return (Tensor) -- output tensor
        """
        return x + self.block(x)

from torchvision.models import vgg19
from torchvision.models._utils import IntermediateLayerGetter
class Vgg19FeatExtrator(nn.Module):
    """ Extracts intermediate features from a pre-trained
    Vgg19 network.
    """
    def __init__(self, return_layers: Dict[str, str], 
                 requires_grad: bool = False) -> None:
        """ Initiates a new feature extrator.

        Given the layers one wants to extract, constructs a new
        feature extrator.

        Parameters:
            return_layers (Dict[str, str]) -- dictionary where aech key is the name of a layer
                                              one wants to extract features from and the respective 
                                              value is the new name that will be given for the extracted 
                                              layer
            requires_grad (bool) -- whether the weights of the vgg needs to be updated or not. 
                                    Default: False.
        """
        super(Vgg19FeatExtrator, self).__init__()
        vgg_features = vgg19(weights='DEFAULT').features
        self.model = IntermediateLayerGetter(vgg_features, return_layers)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> List[Tensor]:
        """ Passes the input through the vgg19 and extracts intermediate features.

        Passes the input through the vgg19 and returns a dict where each key is the new
        name given to the layer and the respective value is the output of such layer.

        Parameters:
            x (Tensor) -- input tensor
        Return (List[Tensor]) -- list of the features extracted, i.e list containing the 
                                 output of each layer specified in the return_layers parameter
        """
        y = [y for y in self.model(x).values()]
        return y
