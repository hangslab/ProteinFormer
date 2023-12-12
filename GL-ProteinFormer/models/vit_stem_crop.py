# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from itertools import repeat
import collections.abc
import math

class ConvStem(nn.Module):
    def __init__(self, in_c=3, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(in_c, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        
        c1 = self.stem(x)
        c1 = self.fc1(c1)


        return c1
        # return outs


def resize_pos_embed(posemb, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    # ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    # print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_grid.shape, hight, width))
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, **kwargs):
        super().__init__()
        norm_layer = None
        flatten = True
        self.embed_dim = embed_dim
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, crop_scale=2, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.patch_size = kwargs['patch_size']
        self.embed_dim = kwargs['embed_dim']
        del self.patch_embed
        # self.patch_embed = PatchEmbed(**kwargs)
        self.input_stem = ConvStem(in_c=4, embed_dim=768)

        # del self.fc_norm
        # del self.pos_embed
        # del self.pre_logits
        del self.head
        self.crop_scale = crop_scale
        num_classes = 13
        self.crop_head = nn.Linear(int(self.crop_scale**2 + 1) * self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        

    def forward_feature(self, x):
        B, inc, H, W = x.shape

        # crop_h = H // crop_scale
        # crop_w = W // crop_scale
        # unfold = nn.Unfold(kernel_size=(crop_h, crop_w), stride=(crop_h, crop_w))
        # # print('x_crop.shape', unfold(x).shape)
        # x_crop = unfold(x).permute(0, 2, 1).reshape(B * int(crop_scale**2), inc, crop_h, crop_w)
        # print('x_crop.shape', x_crop.shape)
        # x = self.patch_embed(x)
        x = self.input_stem(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_up = resize_pos_embed(self.pos_embed, hight=H//self.patch_size, width=W//self.patch_size)
        x = x + pos_embed_up
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            # print('x.shape', x.shape)
            x = self.norm(x)
            # print('out_x.shape', x.shape)
            # outcome = x[:, 1:, :]
            outcome = x[:, 0]
            # outcome = x
            # outcome = self.head(outcome)

        return outcome
    
    def forward(self, x):
        x = 2 * (x / 255.0) - 1.0
        x = x.contiguous()

        B, inc, H, W = x.shape
        crop_scale = self.crop_scale
        crop_h = H // crop_scale
        crop_w = W // crop_scale
        unfold = nn.Unfold(kernel_size=(crop_h, crop_w), stride=(crop_h, crop_w))
        # print('x_crop.shape', unfold(x).shape)
        x_crop = unfold(x).permute(0, 2, 1).reshape(B * int(crop_scale**2), inc, crop_h, crop_w)

        x = self.forward_feature(x)
        x_crop = self.forward_feature(x_crop).reshape(B, -1)

        # print('x', x.shape)
        # print('x_crop', x_crop.shape)
        x = torch.cat((x, x_crop), dim=-1)
        x = self.crop_head(x)
        # x = x.transpose(1, 2).reshape(B, self.embed_dim, H//self.patch_size, W//self.patch_size)
        return x, x


def vit_base_patch16_stem_crop(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16_stem_crop(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14_stem_crop(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model