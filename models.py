import math
import time
from argparse import Namespace
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

import dataset
import utils
from layers import ConvBlock, InteractionNet, GatedGAT, PredictBlock


class PIGNet(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [GatedGAT(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.interaction_net:
            self.interaction_net = nn.ModuleList(
                [InteractionNet(args.dim_gnn) for _ in range(args.n_gnn)]
            )

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Tanh(),
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))

    def cal_hbond(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hbond_coeff * self.hbond_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_hydrophobic(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_vdw_interaction(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        ligand_valid: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(1, 1, target_valid.size(1))
        target_valid_ = target_valid.unsqueeze(1).repeat(1, ligand_valid.size(1), 1)
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )

        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm_0[dm_0 < 0.0001] = 1
        N = self.args.vdw_N
        vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.args.max_vdw_interaction - self.args.min_vdw_interaction)
        A = A + self.args.min_vdw_interaction

        energy = vdw_term1 + vdw_term2
        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_
        energy = A * energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float
    ) -> Tensor:
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(
        self, sample: Dict[str, Any], DM_min: float = 0.5, cal_der_loss: bool = False
    ) -> Tuple[Tensor]:
        (
            ligand_h,
            ligand_adj,
            target_h,
            target_adj,
            interaction_indice,
            ligand_pos,
            target_pos,
            rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_valid,
            target_valid,
            ligand_non_metal,
            target_non_metal,
            _,
            _,
        ) = sample.values()

        # feature embedding
        ligand_h = self.node_embedding(ligand_h)
        target_h = self.node_embedding(target_h)

        # distance matrix
        ligand_pos.requires_grad = True
        dm = self.cal_distance_matrix(ligand_pos, target_pos, DM_min)

        # GatedGAT propagation
        for idx in range(len(self.gconv)):
            ligand_h = self.gconv[idx](ligand_h, ligand_adj)
            target_h = self.gconv[idx](target_h, target_adj)
            ligand_h = F.dropout(
                ligand_h, training=self.training, p=self.args.dropout_rate
            )
            target_h = F.dropout(
                target_h, training=self.training, p=self.args.dropout_rate
            )

        # InteractionNet propagation
        if self.args.interaction_net:
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for idx in range(len(self.interaction_net)):
                new_ligand_h = self.interaction_net[idx](
                    ligand_h,
                    target_h,
                    adj12,
                )
                new_target_h = self.interaction_net[idx](
                    target_h,
                    ligand_h,
                    adj12.permute(0, 2, 1),
                )
                ligand_h, target_h = new_ligand_h, new_target_h
                ligand_h = F.dropout(
                    ligand_h, training=self.training, p=self.args.dropout_rate
                )
                target_h = F.dropout(
                    target_h, training=self.training, p=self.args.dropout_rate
                )

        # concat features
        h1_ = ligand_h.unsqueeze(2).repeat(1, 1, target_h.size(1), 1)
        h2_ = target_h.unsqueeze(1).repeat(1, ligand_h.size(1), 1, 1)
        h_cat = torch.cat([h1_, h2_], -1)

        # compute energy component
        energies = []

        # vdw interaction
        vdw_energy = self.cal_vdw_interaction(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
        )
        energies.append(vdw_energy)

        # hbond interaction
        hbond = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 0],
        )
        energies.append(hbond)

        # metal interaction
        metal = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 1],
        )
        energies.append(metal)

        # hydrophobic interaction
        hydrophobic = self.cal_hydrophobic(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 2],
        )
        energies.append(hydrophobic)

        energies = torch.cat(energies, -1)
        # rotor penalty
        if not self.args.no_rotor_penalty:
            energies = energies / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        # derivatives
        if cal_der_loss:
            gradient = torch.autograd.grad(
                energies.sum(), ligand_pos, retain_graph=True, create_graph=True
            )[0]
            der1 = torch.pow(gradient.sum(1), 2).mean()
            der2 = torch.autograd.grad(
                gradient.sum(), ligand_pos, retain_graph=True, create_graph=True
            )[0]
            der2 = -der2.sum(1).sum(1).mean()
        else:
            der1 = torch.zeros_like(energies).sum()
            der2 = torch.zeros_like(energies).sum()

        return energies, der1, der2


class GNN(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [GatedGAT(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.interaction_net:
            self.interaction_net = nn.ModuleList(
                [InteractionNet(args.dim_gnn) for _ in range(args.n_gnn)]
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

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float
    ) -> Tensor:
        ligand_pos_ = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        target_pos_ = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(ligand_pos_ - target_pos_, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(
        self, sample: Dict[str, Any], DM_min: float = 0.5, cal_der_loss: bool = False
    ) -> Tuple[Tensor]:
        (
            ligand_h,
            ligand_adj,
            target_h,
            target_adj,
            interaction_indice,
            ligand_pos,
            target_pos,
            rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_valid,
            target_valid,
            ligand_non_metal,
            target_non_metal,
            _,
            _,
        ) = sample.values()

        # feature embedding
        ligand_h = self.node_embedding(ligand_h)
        target_h = self.node_embedding(target_h)

        # distance matrix
        dm = self.cal_distance_matrix(ligand_pos, target_pos, DM_min)

        # GatedGAT propagation
        for idx in range(len(self.gconv)):
            ligand_h = self.gconv[idx](ligand_h, ligand_adj)
            target_h = self.gconv[idx](target_h, target_adj)
            ligand_h = F.dropout(
                ligand_h, training=self.training, p=self.args.dropout_rate
            )
            target_h = F.dropout(
                target_h, training=self.training, p=self.args.dropout_rate
            )

        # InteractionNet propagation
        if self.args.interaction_net:
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for idx in range(len(self.interaction_net)):
                new_ligand_h = self.interaction_net[idx](
                    ligand_h,
                    target_h,
                    adj12,
                )
                new_target_h = self.interaction_net[idx](
                    target_h,
                    ligand_h,
                    adj12.permute(0, 2, 1),
                )
                ligand_h, target_h = new_ligand_h, new_target_h
                ligand_h = F.dropout(
                    ligand_h, training=self.training, p=self.args.dropout_rate
                )
                target_h = F.dropout(
                    target_h, training=self.training, p=self.args.dropout_rate
                )

        # concat features
        ligand_h = (ligand_h * ligand_valid.unsqueeze(-1)).sum(1)
        target_h = (target_h * target_valid.unsqueeze(-1)).sum(1)
        h_cat = torch.cat((ligand_h, target_h), -1)

        # compute energy
        energy = self._linear(h_cat, self.predict, nn.ReLU())

        # derivatives (not used)
        der1 = torch.zeros_like(energy).sum()
        der2 = torch.zeros_like(energy).sum()

        return energy, der1, der2

    @staticmethod
    def _linear(
        tensor: Tensor, layers: List[nn.Module], act: Optional[nn.Module] = None
    ) -> Tensor:
        for i, layer in enumerate(layers):
            tensor = layer(tensor)
            if act != None and i != len(layers) - 1:
                tensor = act(tensor)

        return tensor


class CNN3D_KDEEP(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
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

    def forward(
        self, sample: Dict[str, Any], DM_min: float = 0.5, cal_der_loss: bool = False
    ) -> Tuple[Tensor]:
        (
            ligand_h,
            ligand_adj,
            target_h,
            target_adj,
            interaciton_indice,
            ligand_pos,
            target_pos,
            rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_valid,
            target_valid,
            ligand_non_metal,
            target_non_metal,
            _,
            _,
        ) = sample.values()

        batch_size = ligand_pos.shape[0]
        lattice = self.get_lattice(
            ligand_pos,
            target_pos,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_h,
            target_h,
            self.args.lattice_dim,
        )

        if self.args.grid_rotation:
            lattice = lattice.detach().cpu().numpy()  # B, 54, 40, 40, 40
            angle = torch.randint(low=0, high=4, size=(3,))
            lattice = np.rot90(lattice, k=angle[0].item(), axes=(2, 3))
            lattice = np.rot90(lattice, k=angle[1].item(), axes=(3, 4))
            lattice = np.rot90(lattice, k=angle[2].item(), axes=(4, 2))
            lattice = torch.from_numpy(lattice.copy()).to(ligand_h.device)

        lattice = self.conv1(lattice)
        lattice = self.fire2_squeeze(lattice)
        lattice1 = self.fire2_expand2(lattice)
        lattice2 = self.fire2_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.fire3_squeeze(lattice)
        lattice1 = self.fire3_expand2(lattice)
        lattice2 = self.fire3_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.fire4_squeeze(lattice)
        lattice1 = self.fire4_expand2(lattice)
        lattice2 = self.fire4_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.max_pooling4(lattice)
        lattice = self.fire5_squeeze(lattice)
        lattice1 = self.fire5_expand2(lattice)
        lattice2 = self.fire5_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.fire6_squeeze(lattice)
        lattice1 = self.fire6_expand2(lattice)
        lattice2 = self.fire6_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.fire7_squeeze(lattice)
        lattice1 = self.fire7_expand2(lattice)
        lattice2 = self.fire7_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.fire8_squeeze(lattice)
        lattice1 = self.fire8_expand2(lattice)
        lattice2 = self.fire8_expand2(lattice)
        lattice = torch.cat([lattice1, lattice2], dim=1)
        lattice = self.avg_pooling8(lattice)

        lattice = lattice.view(lattice.shape[0], -1)
        energy = self.linear(lattice)

        der1 = torch.zeros_like(energy).sum()
        der2 = torch.zeros_like(energy).sum()

        return energy, der1, der2

    def get_lattice(
        self,
        ligand_pos: Tensor,
        target_pos: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        ligand_h: Tensor,
        target_h: Tensor,
        lattice_dim: int,
    ) -> Tensor:
        n_features = ligand_h.shape[-1]
        device = ligand_pos.device
        batch_size = ligand_pos.size(0)

        lattice_size = int(lattice_dim / self.args.scaling)
        lattice = torch.zeros(
            batch_size,
            lattice_size,
            lattice_size,
            lattice_size,
            n_features,
        )
        ligand_nonzero_pos = (ligand_pos.sum(-1) == 0).unsqueeze(-1)
        ligand_nonzero_pos_max = (ligand_nonzero_pos * -1e10).to(device)
        ligand_nonzero_pos_min = (ligand_nonzero_pos * 1e10).to(device)
        batch_max = torch.max(ligand_pos + ligand_nonzero_pos_max, dim=1)[0]
        batch_min = torch.min(ligand_pos + ligand_nonzero_pos_min, dim=1)[0]

        batch_diff = batch_max - batch_min
        sub = ((batch_min + batch_diff / 2)).unsqueeze(1)
        lattice = lattice.to(device)

        ligand_moved_pos = (ligand_pos - sub) + lattice_dim / 2
        target_moved_pos = (target_pos - sub) + lattice_dim / 2

        grid = torch.zeros([lattice_size, lattice_size, lattice_size])
        grid = torch.transpose(torch.stack(torch.where(grid == 0)), 0, 1)
        grid = grid * self.args.scaling
        grid = grid.to(device)

        ligand_sum = torch.zeros(
            batch_size,
            lattice_size,
            lattice_size,
            lattice_size,
            n_features,
        ).to(device)
        for idx in range(ligand_moved_pos.size(1)):
            ligand_pos_elem = ligand_moved_pos[:, idx, :]
            ligand_h_elem = ligand_h[:, idx, :]
            ligand_vdw_radius = ligand_vdw_radii[:, idx]
            ligand_moved_pos_elem = ligand_pos_elem.unsqueeze(1).repeat(
                1, grid.size(0), 1
            )
            ligand_grid = grid.unsqueeze(0).repeat(ligand_pos_elem.size(0), 1, 1)
            ligand_dist = torch.sqrt(
                torch.pow(ligand_moved_pos_elem - ligand_grid, 2).sum(-1)
            )
            ligand_coeff = 1 - torch.exp(
                -torch.pow(ligand_vdw_radius.unsqueeze(-1) / ligand_dist, 12)
            )
            ligand_coeff = ligand_coeff.view(
                -1, lattice_size, lattice_size, lattice_size
            )
            ligand_h_elem = ligand_h_elem.unsqueeze(1).repeat(1, lattice_size, 1)
            ligand_h_elem = ligand_h_elem.unsqueeze(1).repeat(1, lattice_size, 1, 1)
            ligand_h_elem = ligand_h_elem.unsqueeze(1).repeat(1, lattice_size, 1, 1, 1)
            ligand_coeff = ligand_h_elem * ligand_coeff.unsqueeze(-1)
            ligand_sum += ligand_coeff

        target_sum = torch.zeros(
            batch_size,
            lattice_size,
            lattice_size,
            lattice_size,
            n_features,
        ).to(device)
        for idx in range(target_moved_pos.size(1)):
            target_pos_elem = target_moved_pos[:, idx, :]
            target_h_elem = target_h[:, idx, :]
            target_vdw_radius = target_vdw_radii[:, idx]
            target_moved_pos_elem = target_pos_elem.unsqueeze(1).repeat(1, grid.size(0), 1)
            targeet_grid = grid.unsqueeze(0).repeat(target_pos_elem.size(0), 1, 1)
            target_dist = torch.sqrt(
                torch.pow(target_moved_pos_elem - targeet_grid, 2).sum(-1)
            )
            target_coeff = 1 - torch.exp(
                -torch.pow(target_vdw_radius.unsqueeze(-1) / target_dist, 12)
            )
            target_coeff = target_coeff.view(
                -1, lattice_size, lattice_size, lattice_size
            )
            target_h_elem = target_h_elem.unsqueeze(1).repeat(1, lattice_size, 1)
            target_h_elem = target_h_elem.unsqueeze(1).repeat(1, lattice_size, 1, 1)
            target_h_elem = target_h_elem.unsqueeze(1).repeat(1, lattice_size, 1, 1, 1)
            target_coeff = target_h_elem * target_coeff.unsqueeze(-1)
            target_sum += target_coeff

        lattice = ligand_sum + target_sum
        lattice = lattice.permute(0, 4, 2, 3, 1)

        return lattice

    def _plot(self, lattice: Tensor, idx: int) -> None:
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
        return

    def _add_act(self, func: nn.Module, act: str = "relu") -> nn.Module:
        func_list = []
        func_list.append(func)
        if act == "relu":
            func_list.append(nn.ReLU())

        return nn.Sequential(*func_list)
