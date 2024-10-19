import os
import math
from os.path import join
from typing import Optional, List
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import data_utils

torch.set_float32_matmul_precision('medium')

SAVED_WEIGHTS_PATH = data_utils.get_directory_path("saved_weights")
BOTTLENECK_IMG_SIZE_DICT = {
    # The width and height of an output from a CNN encoder (will be flattened to get latent_dim).
    "mnist": 2,
    "shapes": 2
}
HIDDEN_DIM_DICT = {
    # The series of channels dims in the encoder. Decoder uses the same list in reverse.
    "mnist": [32, 64, 128],
    "shapes": [32, 64, 128, 256]
}
LATENT_DIM_DICT = {
    # The final output from the encoder is shaped [B, latent_dim].
    "mnist": 100,
    "shapes": 100
}


def check_inputs(dataset_information: data_utils.DatasetInformationClass):
    in_channels = dataset_information.number_of_channels
    hidden_dims = HIDDEN_DIM_DICT[dataset_information.name]
    latent_dim = LATENT_DIM_DICT[dataset_information.name]
    img_size = dataset_information.image_size
    latent_size = BOTTLENECK_IMG_SIZE_DICT[dataset_information.name]
    assert math.log2(latent_size).is_integer(), "bottleneck size must be power of 2"
    assert latent_dim % (latent_size ** 2) == 0, f"latent dim {latent_dim} must be divisible by the " \
                                                 f"width*height of the bottleneck: {latent_size ** 2}"
    last_channels = latent_dim // (latent_size ** 2)
    # number of convs to reach (latent_size x latent_size) image:
    num_convs = math.ceil(math.log2(img_size)) - math.ceil(math.log2(latent_size))
    assert len(hidden_dims) + 1 == num_convs, f"data with image size {img_size} needs {num_convs} layers to " \
                                              f"reach ({latent_size} x {latent_size}), but given {len(hidden_dims)+1=}"
    return in_channels, hidden_dims, latent_dim, latent_size, last_channels


def DoubleConv(in_channels, out_channels, mid_channels=None):
    """{convolution -> [BatchNorm] -> ReLU} * 2
    This doesn't change the height and width of the input."""
    if mid_channels is None:
        mid_channels = out_channels
    double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return double_conv


class ConvDeconv_Encoder(nn.Module):
    """
    {Convolution -> BatchNorm -> LeakyReLU}*n -> convolution
    The CNN downsizes the input without any additional pooling functions.
    """

    def __init__(self, dataset_information: data_utils.DatasetInformationClass):
        super().__init__()
        in_channels, hidden_dims, latent_dim, latent_size, last_channels = check_inputs(dataset_information)
        self.latent_dim = latent_dim
        modules = []
        for in_dim, out_dim in pairwise([in_channels] + hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(),
                )
            )
        modules.append(nn.Conv2d(hidden_dims[-1], last_channels, kernel_size=3, stride=2, padding=1))
        self.encoder = nn.Sequential(*modules)
        # self.modules_list = nn.ModuleList(modules)

    def forward(self, img):
        # img: [B, in_channels, img_size, img_size]
        img = self.encoder(img)  # [B, out_channels, latent_size, latent_size]
        img = torch.flatten(img, start_dim=1)   # [B, latent_dim]
        return img


class ConvDeconv_Decoder(nn.Module):
    """
        {ConvTranspose -> BatchNorm -> LeakyReLU}*n -> ConvTranspose -> LeakyReLU -> Conv
        """

    def __init__(self, dataset_information: data_utils.DatasetInformationClass):
        super().__init__()
        final_channels, hidden_dims, _, latent_size, latent_channels = check_inputs(dataset_information)
        hidden_dims = list(reversed(hidden_dims))
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        modules = []
        for in_dim, out_dim in pairwise([self.latent_channels] + hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(),
                    # DoubleConv(out_dim, out_dim)
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], final_channels, kernel_size=3, padding=1),
            # nn.Sigmoid()  # TODO: ablate
        )
        # the decoder returns image sizes which are a power of 2. If the original image size isn't, we add this layer.
        self.orig_size = dataset_information.image_size
        decoder_size = 2 ** math.ceil(math.log2(self.orig_size))
        if not math.log2(self.orig_size).is_integer():
            c = dataset_information.number_of_channels
            self.size_fix_layer = nn.Linear(c * decoder_size ** 2, c * self.orig_size ** 2)
        else:
            self.size_fix_layer = None

    def forward(self, img):
        # img: [B, latent_dim]
        img = img.view(img.size(0), self.latent_channels, self.latent_size, self.latent_size)
        img = self.decoder(img)
        img = self.final_layer(img)
        if self.size_fix_layer is not None:
            b, c, size, _ = img.shape
            correct_size = self.orig_size
            img = torch.flatten(img, start_dim=1)
            img = self.size_fix_layer(img)
            img = img.view(b, c, correct_size, correct_size)
        return img


class ConvPool_Encoder(nn.Module):
    """
    {DoubleConv -> MaxPool}*n -> Conv
    """

    def __init__(self, dataset_information: data_utils.DatasetInformationClass):
        super().__init__()
        in_channels, hidden_dims, latent_dim, latent_size, last_channels = check_inputs(dataset_information)
        self.latent_dim = latent_dim
        self.conv_pools = nn.ModuleList()
        for in_dim, out_dim in pairwise([in_channels] + hidden_dims + [last_channels]):
            self.conv_pools.append(nn.Sequential(
                DoubleConv(in_dim, out_dim, mid_channels=(in_dim + out_dim) // 2),
                nn.MaxPool2d(2, return_indices=True, ceil_mode=True)
            ))
        self.out_conv = nn.Conv2d(last_channels, last_channels, kernel_size=1)

    def forward(self, img):
        # img: [B, in_channels, img_size, img_size]
        pool_indices_list = []
        for conv_pool in self.conv_pools:
            img, pool_indices = conv_pool(img)
            pool_indices_list.append(pool_indices)
        img = self.out_conv(img)
        img = torch.flatten(img, start_dim=1)
        return img, pool_indices_list


class ConvPool_Decoder(nn.Module):
    """
        {DoubleConv -> MaxUnpool}*n -> DoubleConv
        """

    def __init__(self, dataset_information: data_utils.DatasetInformationClass):
        super().__init__()
        final_channels, hidden_dims, _, latent_size, latent_channels = check_inputs(dataset_information)
        hidden_dims = list(reversed(hidden_dims))
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.convs = nn.ModuleList()
        for in_dim, out_dim in pairwise([latent_channels] + hidden_dims + [final_channels]):
            self.convs.append(DoubleConv(in_dim, out_dim, mid_channels=(in_dim + out_dim) // 2))
        self.final_layer = nn.Sequential(
            DoubleConv(final_channels, final_channels),
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            # nn.Sigmoid()  # TODO: ablate
        )

    def forward(self, x):
        img, pool_indices_list = x
        assert len(pool_indices_list) == len(self.convs)
        img = img.view(img.size(0), self.latent_channels, self.latent_size, self.latent_size)
        for conv, indices in zip(self.convs, reversed(pool_indices_list)):
            img = F.max_unpool2d(img, indices, kernel_size=(2, 2))
            img = conv(img)
        img = self.final_layer(img)
        return img


def initialize_encoder(dataset_information: data_utils.DatasetInformationClass or str, architecture_type: str):
    if isinstance(dataset_information, str):
        dataset_information = data_utils.DatasetInformationDict[dataset_information]
    encoder_dict = {
        'conv_deconv': ConvDeconv_Encoder,
        'conv_pool': ConvPool_Encoder
    }
    encoder = encoder_dict[architecture_type](dataset_information)
    return encoder


def initialize_decoder(dataset_information: data_utils.DatasetInformationClass or str, architecture_type: str):
    if isinstance(dataset_information, str):
        dataset_information = data_utils.DatasetInformationDict[dataset_information]
    decoder_dict = {
        'conv_deconv': ConvDeconv_Decoder,
        'conv_pool': ConvPool_Decoder
    }
    decoder = decoder_dict[architecture_type](dataset_information)
    return decoder
