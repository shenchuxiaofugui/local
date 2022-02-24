#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 15:56
# @Author  : Eric Ching
from .unet import UNet3D, UnetVAE3D


def build_model(model_name, input_shape):
    if model_name== 'unet-vae':
        model = UnetVAE3D(input_shape,
                          in_channels=4,
                          out_channels=3,
                          init_channels=16,
                          p=0.2)
    else:
        model = UNet3D(input_shape,
                       in_channels=4,
                       out_channels=3,
                       init_channels=16,
                       p=0.2)

    return model
