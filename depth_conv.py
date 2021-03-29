# Implementation by Marcin Rybok
#
# Source: 
# https://openaccess.thecvf.com/content_ECCV_2018/papers/Weiyue_Wang_Depth-aware_CNN_for_ECCV_2018_paper.pdf
#
# Useful resources on manual implementation of convolutions:
# https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
# https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf
# https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=unfold#torch.nn.Unfold

import math
import torch
import torch.nn as nn

from torch.nn.functional import unfold, pad

class DepthConv2d(nn.Module):
    
    # Current limitations:
    # - Everything is square (input, kernels, padding, etc...)
    # - kernels must be square-shaped, so kernel_size must be int
    # - please use odd kernel_size (makes more sense as even kernel_size does not have well 
    #   defined center pixel)
    # - padding is applied equally on all 4 sides of the image
    # - alpha is the main hyper-parameter of this layer, check the source for its meaning.
    def __init__(self, in_channels, out_channels, kernel_size, alpha=8.3, stride=1, padding=0, dilation=1, bias=True):
        super(DepthConv2d, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.alpha        = alpha
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        self.reset_parameters()
        
    # Initialize or reset layer parameters. 
    # Copy-pasted from PyTorch convolution implementation:
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound     = 1 / math.sqrt(fan_in)
            
            nn.init.uniform_(self.bias, -bound, bound)
        
    # image should be (batch_size, in_channels, input_size, input_size)  (***Please ensure square input***)
    # depth should be (batch_size,           1, input_size, input_size)  (***Please ensure square input***)
    def forward(self, image, depth):
        im2col_depth = depth.repeat(1, self.in_channels, 1, 1)
        batch_size   = image.shape[0]
        
        # Algorithm taken from: https://arxiv.org/pdf/1603.07285.pdf page 28
        output_size = image.shape[-1] + 2 * self.padding - self.kernel_size 
        output_size = output_size - (self.kernel_size - 1) * (self.dilation - 1)
        output_size = int(output_size / self.stride) + 1
        
        # Vectorize inputs
        im2col_input = unfold(image,        self.weight.shape[2:], self.dilation, self.padding, self.stride).transpose(1, 2)
        im2col_depth = unfold(im2col_depth, self.weight.shape[2:], self.dilation, self.padding, self.stride).transpose(1, 2)
        
        # Compute pixel-wise similarity to the central pixel
        centres    = im2col_depth[:, :, int(im2col_depth.shape[2] / 2)]
        centres    = centres.unsqueeze(-1).repeat(1, 1, im2col_depth.shape[2])
        similarity = torch.exp(torch.abs(im2col_depth - centres) * -self.alpha)
        
        # Apply the convolution adjusted by depth similarity
        im2col_input = torch.multiply(im2col_input, similarity)
        im2col_out   = im2col_input.matmul(self.weight.view(self.weight.size(0), -1).t())
        
        if self.bias is not None:
            im2col_out += self.bias
            
        # Reshape result from vectorized to normal form
        im2col_out = im2col_out.transpose(1,2)
        output     = im2col_out.view(batch_size, self.out_channels, output_size, output_size)
        
        return output