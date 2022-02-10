import math
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

import dataset
import utils
from layers import ConvBlock, EdgeConv, GAT_gate, MultiHeadAttention, PredictBlock


class DTIHarmonic(nn.Module):
    def __init__(self, args):
        super(DTIHarmonic, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [GAT_gate(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.edgeconv:
            num_filter = int(10.0 / args.filter_spacing) + 1
            self.filter_center = torch.Tensor(
                [args.filter_spacing * i for i in range(num_filter)]
            )
            self.filter_gamma = args.filter_gamma
            self.edgeconv = nn.ModuleList(
                [EdgeConv(num_filter, args.dim_gnn) for _ in range(args.n_gnn)]
            )
        self.num_interaction_type = len(dataset.interaction_types)

        self.cal_coolomb_interaction_A = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.cal_coolomb_interaction_N = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.vina_hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.vina_hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))
        self.intercept = nn.Parameter(torch.tensor([0.0]))

    def cal_intercept(self, h, valid1, valid2, dm):
        valid1_repeat = valid1.unsqueeze(2).repeat(1, 1, valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1, valid1.size(1), 1)
        C1 = self.cal_intercept_A(h).squeeze(-1) * 0.01
        C2 = self.cal_intercept_B(h).squeeze(-1) * 0.1 + 0.1
        retval = C1 * torch.exp(-torch.pow(C2 * dm, 2))
        retval = retval * valid1_repeat * valid2_repeat
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_coolomb_interaction(self, dm, h, charge1, charge2, valid1, valid2):
        charge1_repeat = charge1.unsqueeze(2).repeat(1, 1, charge2.size(1))
        charge2_repeat = charge2.unsqueeze(1).repeat(1, charge1.size(1), 1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1, 1, valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1, valid1.size(1), 1)
        A = self.cal_coolomb_interaction_A(h).squeeze(-1)
        # A = self.coolomb_coeff*self.coolomb_coeff
        N = self.cal_coolomb_interaction_N(h).squeeze(-1) * 2 + 1
        charge12 = charge1_repeat * charge2_repeat
        energy = A * charge12 * torch.pow(1 / dm, N)
        energy = energy * valid1_repeat * valid2_repeat
        energy = energy.clamp(min=-100, max=100)
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def vina_steric(self, dm, h, vdw_radius1, vdw_radius2, valid1, valid2):
        valid1_repeat = valid1.unsqueeze(2).repeat(1, 1, valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1, valid1.size(1), 1)
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2).repeat(1, 1, vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1).repeat(1, vdw_radius1.size(1), 1)
        dm_0 = vdw_radius1_repeat + vdw_radius2_repeat
        dm = dm - dm_0
        g1 = torch.exp(-torch.pow(dm / 0.5, 2)) * -0.0356
        g2 = torch.exp(-torch.pow((dm - 3) / 2, 2)) * -0.00516
        repulsion = dm * dm * 0.84
        zero_vec = torch.zeros_like(repulsion)
        repulsion = torch.where(dm > 0, zero_vec, repulsion)
        retval = g1 + g2 + repulsion
        retval = retval * valid1_repeat * valid2_repeat
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def vina_hbond(self, dm, h, vdw_radius1, vdw_radius2, A):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2).repeat(1, 1, vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1).repeat(1, vdw_radius1.size(1), 1)
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = vdw_radius1_repeat + vdw_radius2_repeat + B
        dm = dm - dm_0
        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)

        coeff = self.vina_hbond_coeff * self.vina_hbond_coeff
        retval = retval * -coeff
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def vina_hydrophobic(self, dm, h, vdw_radius1, vdw_radius2, A):
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2).repeat(1, 1, vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1).repeat(1, vdw_radius1.size(1), 1)
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = vdw_radius1_repeat + vdw_radius2_repeat + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -self.vina_hydrophobic_coeff * self.vina_hydrophobic_coeff
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_vdw_interaction(
        self, dm, h, vdw_radius1, vdw_radius2, vdw_epsilon, vdw_sigma, valid1, valid2
    ):
        valid1_repeat = valid1.unsqueeze(2).repeat(1, 1, valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1, valid1.size(1), 1)
        vdw_radius1_repeat = vdw_radius1.unsqueeze(2).repeat(1, 1, vdw_radius2.size(1))
        vdw_radius2_repeat = vdw_radius2.unsqueeze(1).repeat(1, vdw_radius1.size(1), 1)

        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = vdw_radius1_repeat + vdw_radius2_repeat + B
        # dm_0 = vdw_sigma
        dm_0[dm_0 < 0.0001] = 1
        N = self.args.vdw_N
        vdw1 = torch.pow(dm_0 / dm, 2 * N)
        vdw2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.args.max_vdw_interaction - self.args.min_vdw_interaction)
        A = A + self.args.min_vdw_interaction

        energy = vdw1 + vdw2
        energy = energy.clamp(max=100)
        energy = energy * valid1_repeat * valid2_repeat
        energy = A * energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def cal_torsion_energy(self, torsion_energy):
        retval = torsion_energy * self.vdw_coeff * self.vdw_coeff
        return retval.unsqueeze(-1)

    def cal_distance_matrix(self, p1, p2, dm_min):
        p1_repeat = p1.unsqueeze(2).repeat(1, 1, p2.size(1), 1)
        p2_repeat = p2.unsqueeze(1).repeat(1, p1.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(self, sample, DM_min=0.5, cal_der_loss=False):
        (
            h1,
            adj1,
            h2,
            adj2,
            A_int,
            dmv,
            _,
            pos1,
            pos2,
            sasa,
            dsasa,
            rotor,
            charge1,
            charge2,
            vdw_radius1,
            vdw_radius2,
            vdw_epsilon,
            vdw_sigma,
            delta_uff,
            valid1,
            valid2,
            no_metal1,
            no_metal2,
            _,
            _,
        ) = sample.values()

        h1 = self.node_embedding(h1)
        h2 = self.node_embedding(h2)

        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1)
            h2 = self.gconv[i](h2, adj2)
            h1 = F.dropout(h1, training=self.training, p=self.args.dropout_rate)
            h2 = F.dropout(h2, training=self.training, p=self.args.dropout_rate)
        pos1.requires_grad = True
        dm = self.cal_distance_matrix(pos1, pos2, DM_min)
        if self.args.edgeconv:
            edge = dm.unsqueeze(-1).repeat(1, 1, 1, self.filter_center.size(-1))
            filter_center = (
                self.filter_center.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(h1.device)
            )

            edge = torch.exp(-torch.pow(edge - filter_center, 2) * self.filter_gamma)
            edge = edge.detach()
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for i in range(len(self.edgeconv)):
                new_h1 = self.edgeconv[i](
                    h1, h2, edge, adj12
                )  # [, n_ligand_atom, n_out_feature(dim_gnn)]
                new_h2 = self.edgeconv[i](
                    h2, h1, edge.permute(0, 2, 1, 3), adj12.permute(0, 2, 1)
                )  # [, n_protein_atom, n_out_feature(dim_gnn)]
                h1, h2 = new_h1, new_h2
                h1 = F.dropout(h1, training=self.training, p=self.args.dropout_rate)
                h2 = F.dropout(h2, training=self.training, p=self.args.dropout_rate)

        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1)
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1)
        h = torch.cat([h1_repeat, h2_repeat], -1)

        retval = []

        # vdw interaction
        E_vdw = self.cal_vdw_interaction(
            dm,
            h,
            vdw_radius1,
            vdw_radius2,
            vdw_epsilon,
            vdw_sigma,
            no_metal1,
            no_metal2,
        )
        retval.append(E_vdw)

        retval.append(self.vina_hbond(dm, h, vdw_radius1, vdw_radius2, A_int[:, 1]))
        retval.append(self.vina_hbond(dm, h, vdw_radius1, vdw_radius2, A_int[:, -1]))
        retval.append(
            self.vina_hydrophobic(dm, h, vdw_radius1, vdw_radius2, A_int[:, -2])
        )
        retval.append(self.cal_torsion_energy(delta_uff))

        retval = torch.cat(retval, -1)
        if not self.args.no_rotor_penalty:
            retval = retval / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        if cal_der_loss:
            minimum_loss = torch.autograd.grad(
                retval.sum(), pos1, retain_graph=True, create_graph=True
            )[0]
            minimum_loss2 = torch.pow(minimum_loss.sum(1), 2).mean()
            minimum_loss3 = torch.autograd.grad(
                minimum_loss.sum(), pos1, retain_graph=True, create_graph=True
            )[0]
            minimum_loss3 = -minimum_loss3.sum(1).sum(1).mean()
        else:
            minimum_loss2 = torch.zeros_like(retval).sum()
            minimum_loss3 = torch.zeros_like(retval).sum()
        return retval, minimum_loss2, minimum_loss3


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [GAT_gate(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.edgeconv:
            num_filter = int(10.0 / args.filter_spacing) + 1
            self.filter_center = torch.Tensor(
                [args.filter_spacing * i for i in range(num_filter)]
            )
            self.filter_gamma = args.filter_gamma
            self.edgeconv = nn.ModuleList(
                [EdgeConv(num_filter, args.dim_gnn) for _ in range(args.n_gnn)]
            )

        if self.training:
            self.predict = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(args.dim_gnn * 2, args.dim_gnn),
                        nn.Dropout(p=args.dropout_rate),
                    ),
                    nn.Sequential(
                        nn.Linear(args.dim_gnn, args.dim_gnn // 2),
                        nn.Dropout(p=args.dropout_rate),
                    ),
                    nn.Sequential(nn.Linear(args.dim_gnn // 2, 1)),
                ]
            )
        else:
            self.predict = nn.ModuleList(
                [
                    nn.Linear(args.dim_gnn * 2, args.dim_gnn),
                    nn.Linear(args.dim_gnn, args.dim_gnn // 2),
                    nn.Linear(args.dim_gnn // 2, 1),
                ]
            )

    def cal_distance_matrix(self, p1, p2, dm_min):
        p1_repeat = p1.unsqueeze(2).repeat(1, 1, p2.size(1), 1)
        p2_repeat = p2.unsqueeze(1).repeat(1, p1.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(self, sample, DM_min=0.5, cal_der_loss=False):
        (
            h1,
            adj1,
            h2,
            adj2,
            A_int,
            dmv,
            _,
            pos1,
            pos2,
            sasa,
            dsasa,
            rotor,
            charge1,
            charge2,
            vdw_radius1,
            vdw_radius2,
            vdw_epsilon,
            vdw_sigma,
            delta_uff,
            valid1,
            valid2,
            no_metal1,
            no_metal2,
            _,
            _,
        ) = sample.values()

        h1 = self.node_embedding(h1)
        h2 = self.node_embedding(h2)

        for i in range(len(self.gconv)):
            h1 = self.gconv[i](h1, adj1)
            h2 = self.gconv[i](h2, adj2)
            h1 = F.dropout(h1, training=self.training, p=self.args.dropout_rate)
            h2 = F.dropout(h2, training=self.training, p=self.args.dropout_rate)

        dm = self.cal_distance_matrix(pos1, pos2, DM_min)
        if self.args.edgeconv:
            edge = dm.unsqueeze(-1).repeat(1, 1, 1, self.filter_center.size(-1))
            filter_center = (
                self.filter_center.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(h1.device)
            )

            edge = torch.exp(-torch.pow(edge - filter_center, 2) * self.filter_gamma)
            edge = edge.detach()
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for i in range(len(self.edgeconv)):
                new_h1 = self.edgeconv[i](
                    h1, h2, edge, adj12
                )  # [, n_ligand_atom, n_out_feature(dim_gnn)]
                new_h2 = self.edgeconv[i](
                    h2, h1, edge.permute(0, 2, 1, 3), adj12.permute(0, 2, 1)
                )  # [, n_protein_atom, n_out_feature(dim_gnn)]
                h1, h2 = new_h1, new_h2
                h1 = F.dropout(h1, training=self.training, p=self.args.dropout_rate)
                h2 = F.dropout(h2, training=self.training, p=self.args.dropout_rate)

        h1_repeat = h1.unsqueeze(2).repeat(1, 1, h2.size(1), 1)
        h2_repeat = h2.unsqueeze(1).repeat(1, h1.size(1), 1, 1)
        valid1_repeat = valid1.unsqueeze(2).repeat(1, 1, valid2.size(1))
        valid2_repeat = valid2.unsqueeze(1).repeat(1, valid1.size(1), 1)
        h1_repeat = h1_repeat * valid1_repeat.unsqueeze(-1)
        h2_repeat = h2_repeat * valid2_repeat.unsqueeze(-1)

        h1 = (h1 * valid1.unsqueeze(-1)).sum(1)  # [, n_out_feature(dim_gnn)]
        h2 = (h2 * valid2.unsqueeze(-1)).sum(1)  # [, n_out_feature(dim_gnn)]
        h = torch.cat((h1, h2), -1)  # [, 2*n_out_feature(dim_gnn)]
        retval = self._linear(h, self.predict, nn.ReLU())

        minimum_loss2 = torch.zeros_like(retval).sum()
        minimum_loss3 = torch.zeros_like(retval).sum()

        return retval, minimum_loss2, minimum_loss3

    @staticmethod
    def _linear(tensor, layers, act=None):
        for i, layer in enumerate(layers):
            tensor = layer(tensor)
            if act != None and i != len(layers) - 1:
                tensor = act(tensor)

        return tensor


class CNN3D(nn.Module):
    def __init__(self, args):
        super(CNN3D, self).__init__()
        self.args = args
        self.size = 20

        # self.conv = ConvBlock(54, 64, args.dropout_rate)
        self.conv = ConvBlock(54, 128, args.dropout_rate)

        # self.predict = PredictBlock(64*40*40*40, 1, args.dropout_rate, True)
        self.predict = PredictBlock(128 * 40 * 40 * 40, 1, args.dropout_rate, True)

    def forward(self, sample, DM_min=0.5, cal_der_loss=False):
        (
            h1,
            adj1,
            h2,
            adj2,
            A_int,
            dmv,
            _,
            pos1,
            pos2,
            sasa,
            dsasa,
            rotor,
            charge1,
            charge2,
            vdw_radius1,
            vdw_radius2,
            vdw_epsilon,
            vdw_sigma,
            delta_uff,
            valid1,
            valid2,
            no_metal1,
            no_metal2,
            _,
            _,
        ) = sample.values()

        batch_size = pos1.shape[0]
        h1 = h1 * valid1.unsqueeze(-1)
        h2 = h2 * valid2.unsqueeze(-1)
        pos1 = pos1 * valid1.unsqueeze(-1)
        pos2 = pos2 * valid2.unsqueeze(-1)
        lattice = self._get_lattice(batch_size, pos1, pos2, h1, h2, self.size)
        lattice = lattice.detach().cpu().numpy()  # B, 54, 40, 40, 40
        angle = torch.randint(low=0, high=4, size=(3,))
        lattice = np.rot90(lattice, k=angle[0].item(), axes=(2, 3))
        lattice = np.rot90(lattice, k=angle[1].item(), axes=(3, 4))
        lattice = np.rot90(lattice, k=angle[2].item(), axes=(4, 2))
        lattice = torch.from_numpy(lattice.copy()).to(h1.device)

        lattice = self.conv(lattice)
        lattice = lattice.view(lattice.shape[0], -1)
        retval = self.predict(lattice)

        minimum_loss2 = torch.zeros_like(retval).sum()
        minimum_loss3 = torch.zeros_like(retval).sum()

        return retval, minimum_loss2, minimum_loss3

    def _get_lattice(self, batch_size, pos1, pos2, h1, h2, lattice_size):
        n_feature = h1.shape[-1]
        ranges = lattice_size * 2
        device = pos1.device
        lattice = torch.zeros(batch_size, ranges, ranges, ranges, n_feature)
        nz_pos1 = (pos1.sum(-1) == 0).unsqueeze(-1)
        nz_pos1_max = nz_pos1 * -1e10
        nz_pos1_min = nz_pos1 * 1e10
        batch_max = torch.max(pos1 + nz_pos1_max.to(device), dim=1)[0]
        batch_min = torch.min(pos1 + nz_pos1_min.to(device), dim=1)[0]
        batch_diff = batch_max - batch_min
        sub = ((batch_min + batch_diff / 2)).unsqueeze(1)
        index1 = ((pos1 - sub + lattice_size / 2) // 0.5).type(torch.IntTensor)  # index
        index2 = ((pos2 - sub + lattice_size / 2) // 0.5).type(torch.IntTensor)  # index
        lattice = lattice.to(device)

        # fill lattice with h1, h2's one-hot vector
        batch_pos_feat1 = zip(index1, h1)
        for i, (batch_pos1, batch_feat1) in enumerate(batch_pos_feat1):
            pos_feat1 = zip(batch_pos1, batch_feat1)
            for (coor1, feature1) in pos_feat1:
                x1, y1, z1 = coor1
                if (
                    x1 < 0
                    or x1 > ranges - 1
                    or y1 < 0
                    or y1 > ranges - 1
                    or z1 < 0
                    or z1 > ranges - 1
                ):
                    continue
                lattice[i][x1][y1][z1] = feature1
        batch_pos_feat2 = zip(index2, h2)
        for j, (batch_pos2, batch_feat2) in enumerate(batch_pos_feat2):
            pos_feat2 = zip(batch_pos2, batch_feat2)
            for (coor2, feature2) in pos_feat2:
                x2, y2, z2 = coor2
                if (
                    x2 < 0
                    or x2 > ranges - 1
                    or y2 < 0
                    or y2 > ranges - 1
                    or z2 < 0
                    or z2 > ranges - 1
                ):
                    continue
                lattice[j][x2][y2][z2] = feature2

        lattice = lattice.permute(0, 4, 2, 3, 1)  # b, f, y, z, x

        return lattice

    def _plot(self, lattice, idx):
        lattice = lattice.permute(0, 4, 2, 3, 1)  # b, f, y, z, x
        lattice_0 = lattice[0].sum(-1)
        lattice_1 = lattice[1].sum(-1)

        voxels_0 = lattice_0 != 0
        voxels_1 = lattice_1 != 0
        voxels = voxels_0 | voxels_1

        colors = np.empty(voxels.shape, dtype=object)
        colors[voxels_0] = "green"
        colors[voxels_1] = "red"
        if lattice.shape[0] > 2:
            lattice_2 = lattice[2].sum(-1)
            lattice_3 = lattice[3].sum(-1)
            voxels_2 = lattice_2 != 0
            voxels_3 = lattice_3 != 0
            voxels = voxels | voxels_2 | voxels_3
            colors[voxels_2] = "yellow"
            colors[voxels_3] = "purple"

        fig = plt.figure(idx)
        ax = fig.gca(projection="3d")
        ax.voxels(voxels, facecolors=colors, edgecolor="k")


class CNN3D_KDEEP(nn.Module):
    def __init__(self, args):
        super(CNN3D_KDEEP, self).__init__()
        self.args = args
        lattice_dim = args.lattice_dim
        scaling = args.scaling
        lattice_size = int(lattice_dim / scaling)
        self.conv1 = self._add_act(nn.Conv3d(54, 96, 2, 2, 0))
        self.fire2_squeeze = self._add_act(nn.Conv3d(96, 16, 3, 1, 1))
        self.fire2_expand1 = self._add_act(nn.Conv3d(16, 64, 3, 1, 1))
        self.fire2_expand2 = self._add_act(nn.Conv3d(16, 64, 3, 1, 1))

        self.fire3_squeeze = self._add_act(nn.Conv3d(128, 16, 3, 1, 1))
        self.fire3_expand1 = self._add_act(nn.Conv3d(16, 64, 3, 1, 1))
        self.fire3_expand2 = self._add_act(nn.Conv3d(16, 64, 3, 1, 1))

        self.fire4_squeeze = self._add_act(nn.Conv3d(128, 32, 3, 1, 1))
        self.fire4_expand1 = self._add_act(nn.Conv3d(32, 128, 3, 1, 1))
        self.fire4_expand2 = self._add_act(nn.Conv3d(32, 128, 3, 1, 1))
        self.max_pooling4 = nn.MaxPool3d(2, 3, 1)

        self.fire5_squeeze = self._add_act(nn.Conv3d(256, 32, 3, 1, 1))
        self.fire5_expand1 = self._add_act(nn.Conv3d(32, 128, 3, 1, 1))
        self.fire5_expand2 = self._add_act(nn.Conv3d(32, 128, 3, 1, 1))

        self.fire6_squeeze = self._add_act(nn.Conv3d(256, 48, 3, 1, 1))
        self.fire6_expand1 = self._add_act(nn.Conv3d(48, 192, 3, 1, 1))
        self.fire6_expand2 = self._add_act(nn.Conv3d(48, 192, 3, 1, 1))

        self.fire7_squeeze = self._add_act(nn.Conv3d(384, 48, 3, 1, 1))
        self.fire7_expand1 = self._add_act(nn.Conv3d(48, 192, 3, 1, 1))
        self.fire7_expand2 = self._add_act(nn.Conv3d(48, 192, 3, 1, 1))

        self.fire8_squeeze = self._add_act(nn.Conv3d(384, 64, 3, 1, 1))
        self.fire8_expand1 = self._add_act(nn.Conv3d(64, 256, 3, 1, 1))
        self.fire8_expand2 = self._add_act(nn.Conv3d(64, 256, 3, 1, 1))

        self.avg_pooling8 = nn.AvgPool3d(3, 2, 0)

        self.linear = nn.Linear(4096, 1)

    def forward(self, sample, DM_min=0.5, cal_der_loss=False):
        (
            h1,
            adj1,
            h2,
            adj2,
            A_int,
            dmv,
            _,
            pos1,
            pos2,
            sasa,
            dsasa,
            rotor,
            charge1,
            charge2,
            vdw_radius1,
            vdw_radius2,
            vdw_epsilon,
            vdw_sigma,
            delta_uff,
            valid1,
            valid2,
            no_metal1,
            no_metal2,
            _,
            _,
        ) = sample.values()

        batch_size = pos1.shape[0]
        lattice = self._get_lattice(
            pos1, pos2, vdw_radius1, vdw_radius2, h1, h2, self.args.lattice_dim
        )

        if self.args.grid_rotation:
            lattice = lattice.detach().cpu().numpy()  # B, 54, 40, 40, 40
            angle = torch.randint(low=0, high=4, size=(3,))
            lattice = np.rot90(lattice, k=angle[0].item(), axes=(2, 3))
            lattice = np.rot90(lattice, k=angle[1].item(), axes=(3, 4))
            lattice = np.rot90(lattice, k=angle[2].item(), axes=(4, 2))
            lattice = torch.from_numpy(lattice.copy()).to(h1.device)
        # print(lattice.shape)

        lattice = self.conv1(lattice)
        # print(lattice.shape)
        lattice = self.fire2_squeeze(lattice)
        lattice1 = self.fire2_expand2(lattice)
        lattice2 = self.fire2_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.fire3_squeeze(lattice)
        lattice1 = self.fire3_expand2(lattice)
        lattice2 = self.fire3_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.fire4_squeeze(lattice)
        lattice1 = self.fire4_expand2(lattice)
        lattice2 = self.fire4_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.max_pooling4(lattice)
        lattice = self.fire5_squeeze(lattice)
        lattice1 = self.fire5_expand2(lattice)
        lattice2 = self.fire5_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.fire6_squeeze(lattice)
        lattice1 = self.fire6_expand2(lattice)
        lattice2 = self.fire6_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.fire7_squeeze(lattice)
        lattice1 = self.fire7_expand2(lattice)
        lattice2 = self.fire7_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.fire8_squeeze(lattice)
        lattice1 = self.fire8_expand2(lattice)
        lattice2 = self.fire8_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        # print(lattice.shape)
        lattice = self.avg_pooling8(lattice)
        # print(lattice.shape)

        lattice = lattice.view(lattice.shape[0], -1)
        retval = self.linear(lattice)
        # print(retval.shape)

        minimum_loss2 = torch.zeros_like(retval).sum()
        minimum_loss3 = torch.zeros_like(retval).sum()

        return retval, minimum_loss2, minimum_loss3

    def _get_lattice(self, pos1, pos2, vr1, vr2, h1, h2, lattice_dim):
        n_feature = h1.shape[-1]
        device = pos1.device
        batch_size = pos1.size(0)

        lattice_size = int(lattice_dim / self.args.scaling)
        lattice = torch.zeros(
            batch_size, lattice_size, lattice_size, lattice_size, n_feature
        )
        nz_pos1 = (pos1.sum(-1) == 0).unsqueeze(-1)
        nz_pos1_max = (nz_pos1 * -1e10).to(device)
        nz_pos1_min = (nz_pos1 * 1e10).to(device)
        batch_max = torch.max(pos1 + nz_pos1_max, dim=1)[0]
        batch_min = torch.min(pos1 + nz_pos1_min, dim=1)[0]

        batch_diff = batch_max - batch_min
        sub = ((batch_min + batch_diff / 2)).unsqueeze(1)
        lattice = lattice.to(device)

        moved_pos1 = (pos1 - sub) + lattice_dim / 2
        moved_pos2 = (pos2 - sub) + lattice_dim / 2

        grid = torch.zeros([lattice_size, lattice_size, lattice_size])
        grid = torch.transpose(torch.stack(torch.where(grid == 0)), 0, 1)
        grid = grid * self.args.scaling
        grid = grid.to(device)

        sum1 = torch.zeros(
            batch_size, lattice_size, lattice_size, lattice_size, n_feature
        ).to(device)
        for i in range(moved_pos1.size(1)):
            pe1 = moved_pos1[:, i, :]
            he1 = h1[:, i, :]
            vre1 = vr1[:, i]
            mp1 = pe1.unsqueeze(1).repeat(1, grid.size(0), 1)
            g1r = grid.unsqueeze(0).repeat(pe1.size(0), 1, 1)
            de1 = torch.sqrt(torch.pow(mp1 - g1r, 2).sum(-1))
            ce1 = 1 - torch.exp(-torch.pow(vre1.unsqueeze(-1) / de1, 12))
            ce1 = ce1.view(-1, lattice_size, lattice_size, lattice_size)
            he1 = he1.unsqueeze(1).repeat(1, lattice_size, 1)
            he1 = he1.unsqueeze(1).repeat(1, lattice_size, 1, 1)
            he1 = he1.unsqueeze(1).repeat(1, lattice_size, 1, 1, 1)
            mul1 = he1 * ce1.unsqueeze(-1)
            sum1 += mul1

        sum2 = torch.zeros(
            batch_size, lattice_size, lattice_size, lattice_size, n_feature
        ).to(device)
        for i in range(moved_pos2.size(1)):
            pe2 = moved_pos2[:, i, :]
            he2 = h2[:, i, :]
            vre2 = vr2[:, i]
            mp2 = pe2.unsqueeze(1).repeat(1, grid.size(0), 1)
            g2r = grid.unsqueeze(0).repeat(pe2.size(0), 1, 1)
            de2 = torch.sqrt(torch.pow(mp2 - g2r, 2).sum(-1))
            ce2 = 1 - torch.exp(-torch.pow(vre2.unsqueeze(-1) / de2, 12))
            ce2 = ce2.view(-1, lattice_size, lattice_size, lattice_size)
            he2 = he2.unsqueeze(1).repeat(1, lattice_size, 1)
            he2 = he2.unsqueeze(1).repeat(1, lattice_size, 1, 1)
            he2 = he2.unsqueeze(1).repeat(1, lattice_size, 1, 1, 1)
            mul2 = he2 * ce2.unsqueeze(-1)
            sum2 += mul2

        lattice = sum1 + sum2
        lattice = lattice.permute(0, 4, 2, 3, 1)

        return lattice

    def _plot(self, lattice, idx):
        lattice = lattice.permute(0, 4, 2, 3, 1)  # b, f, y, z, x
        lattice_0 = lattice[0].sum(-1)
        lattice_1 = lattice[1].sum(-1)

        voxels_0 = lattice_0 != 0
        voxels_1 = lattice_1 != 0
        voxels = voxels_0 | voxels_1

        colors = np.empty(voxels.shape, dtype=object)
        colors[voxels_0] = "green"
        colors[voxels_1] = "red"
        if lattice.shape[0] > 2:
            lattice_2 = lattice[2].sum(-1)
            lattice_3 = lattice[3].sum(-1)
            voxels_2 = lattice_2 != 0
            voxels_3 = lattice_3 != 0
            voxels = voxels | voxels_2 | voxels_3
            colors[voxels_2] = "yellow"
            colors[voxels_3] = "purple"

        fig = plt.figure(idx)
        ax = fig.gca(projection="3d")
        ax.voxels(voxels, facecolors=colors, edgecolor="k")

    def _add_act(self, func, act="relu"):
        func_list = []
        func_list.append(func)
        if act == "relu":
            func_list.append(nn.ReLU())

        return nn.Sequential(*func_list)
