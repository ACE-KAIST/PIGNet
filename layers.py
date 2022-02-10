import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedGAT(nn.Module):
    def __init__(self, n_in_feature: int, n_out_feature: int):
        super().__init__()

        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_in_feature + n_out_feature, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        h = self.W(x)
        e = torch.einsum("ijl,ikl->ijk", (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 1e-6, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = attention * adj
        h_prime = F.relu(torch.einsum("aij,ajk->aik", (attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(
            1, 1, x.size(-1)
        )
        new_x = coeff * x + (1 - coeff) * h_prime
        return new_x


class InteractionNet(nn.Module):
    def __init__(self, n_atom_feature: int):
        super().__init__()

        self.W = nn.Linear(n_atom_feature, n_atom_feature)
        self.M = nn.Linear(n_atom_feature, n_atom_feature)
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)

    def forward(self, x1: Tensor, x2: Tensor, valid_edge: Tensor) -> Tensor:
        new_edge = x2.unsqueeze(1).repeat(1, x1.size(1), 1, 1)

        m1 = self.W(x1)
        m2 = (self.M(new_edge) * valid_edge.unsqueeze(-1)).max(2)[0]
        x_cat = F.relu(m1 + m2)
        feature_size = x_cat.size(-1)
        x_cat = self.C(x_cat.reshape(-1, feature_size), x1.reshape(-1, feature_size))
        x_cat = x_cat.reshape(x1.size(0), x1.size(1), x1.size(2))
        return x_cat


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        do: float = 0.0,
        stride: int = 1,
        kernel: int = 3,
        pad: int = 1,
        bn: bool = True,
    ):
        super().__init__()

        self.block = []
        self.block.append(nn.Conv3d(in_feature, out_feature, kernel, stride, pad))
        if bn:
            self.block.append(nn.BatchNorm3d(out_feature))
        self.block.append(nn.ReLU())
        if do != 0:
            self.block.append(nn.Dropout3d(p=do))
        self.block = nn.Sequential(*self.block)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PredictBlock(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        dropout: float,
        is_last: bool,
    ):
        super().__init__()

        self.block = []
        self.block.append(nn.Linear(in_feature, out_feature))
        if not is_last:
            self.block.append(nn.Dropout(p=dropout))
            self.block.append(nn.ReLU())
        self.block = nn.Sequential(*self.block)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
