###############################################################################
# The code for the kw-mlp model is mostly adapted from lucidrains/g-mlp-pytorch
###############################################################################
# MIT License
#
# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################


import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from random import randrange


# helpers

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, act = nn.Identity(), init_eps = 1e-3):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)

        self.act = act

        init_eps /= dim_seq
        nn.init.uniform_(self.proj.weight, -init_eps, init_eps)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        gate = F.conv1d(gate, weight, bias)

        return self.act(gate) * res


class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        act = nn.Identity()
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return x


class KW_MLP_SingleHeadLang(nn.Module):
    """Keyword-MLP for language identification."""
    
    def __init__(
        self,
        input_res = [40, 98],
        patch_res = [40, 1],
        num_langs = 4,
        dim = 64,
        depth = 12,
        ff_mult = 4,
        channels = 1,
        prob_survival = 0.9,
        pre_norm = False,
        **kwargs
    ):
        super().__init__()
        image_height, image_width = input_res
        patch_height, patch_width = patch_res
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        P_Norm = PreNorm if pre_norm else PostNorm
        
        dim_ff = dim * ff_mult

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(channels * patch_height * patch_width, dim)
        )

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList(
            [Residual(P_Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=num_patches))) for i in range(depth)]
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_langs)
        )

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)
