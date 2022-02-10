import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNN(torch.nn.Module):
    def __init__(self, n_edge_feature, n_atom_feature):
        super(MPNN, self).__init__()

        self.W = nn.Linear(n_atom_feature, n_atom_feature)
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            # nn.Linear(n_feature, 3*n_feature),
            # nn.ReLU(),
            # nn.Linear(3*n_feature, 1*n_feature),
            # nn.ReLU(),
            nn.Linear(1 * n_edge_feature, n_atom_feature * n_atom_feature),
            # nn.ReLU(),
        )
        self.A = nn.Parameter(torch.zeros(size=(n_atom_feature, n_atom_feature)))

    def forward(self, x1, x2, edge):
        message_matrix = self.cal_message(edge)

        message_matrix = message_matrix.view(
            edge.size(0), edge.size(1), edge.size(2), x1.size(-1), x1.size(-1)
        )
        x_repeat = x2.unsqueeze(1).repeat(1, x1.size(1), 1, 1).unsqueeze(-2)

        message = torch.einsum("abcde,abcef->abcdf", (x_repeat, message_matrix))
        message = message.squeeze(-2)
        message = message.sum(2).squeeze()

        reshaped_message = message.view(-1, x1.size(-1))
        reshaped_x = x1.view(-1, x1.size(-1))
        retval = self.C(reshaped_message, reshaped_x)
        retval = retval.view(x1.size(0), x1.size(1), x1.size(2))
        return retval


class EdgeConv(torch.nn.Module):
    def __init__(self, n_edge_feature, n_atom_feature):
        super(EdgeConv, self).__init__()

        self.W = nn.Linear(n_atom_feature, n_atom_feature)
        # self.M = nn.Linear(n_atom_feature, n_atom_feature)
        self.M = nn.Linear(n_atom_feature, n_atom_feature)
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)

    def forward(self, x1, x2, edge, valid_edge):
        new_edge = x2.unsqueeze(1).repeat(1, x1.size(1), 1, 1)
        retval = 0

        m1 = self.W(x1)
        m2 = (self.M(new_edge) * valid_edge.unsqueeze(-1)).max(2)[0]
        retval = F.relu(m1 + m2)
        feature_size = retval.size(-1)
        retval = self.C(retval.reshape(-1, feature_size), x1.reshape(-1, feature_size))
        retval = retval.reshape(x1.size(0), x1.size(1), x1.size(2))
        return retval


class IntraNet(torch.nn.Module):
    def __init__(self, n_atom_feature, n_edge_feature):
        super(IntraNet, self).__init__()

        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            nn.Linear(n_atom_feature * 2 + n_edge_feature, n_atom_feature),
            nn.ReLU(),
            nn.Linear(n_atom_feature, n_atom_feature * 3),
            nn.ReLU(),
            nn.Linear(n_atom_feature * 3, n_atom_feature),
        )

    def forward(self, edge, adj, x):
        h1 = x.unsqueeze(1).repeat(1, x.size(1), 1, 1)
        h2 = x.unsqueeze(2).repeat(1, 1, x.size(1), 1)

        concat = torch.cat([h1, h2, edge], -1)
        message = self.cal_message(concat)
        message = message * adj.unsqueeze(-1).repeat(1, 1, 1, message.size(-1))
        message = message.sum(2).squeeze()

        # norm = torch.norm(message, p=2, dim=-1, keepdim=True)
        # message = message.div(norm.expand_as(message))
        norm = adj.sum(2, keepdim=True)
        message = message.div(norm.expand_as(message) + 1e-6)

        reshaped_message = message.view(-1, x.size(-1))
        reshaped_x = x.view(-1, x.size(-1))
        retval = self.C(reshaped_message, reshaped_x)
        retval = retval.view(x.size(0), x.size(1), x.size(2))

        return retval


class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        # self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_in_feature + n_out_feature, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        """
        -graph attention gate
        h = WX [n_atom * n_out_feature]
        e = WXA * tr(WX) + tr(WXA * tr(WX)) [n_atom * n_atom]
        attention = Softmax(torch.where(adj > 1e-6, e, zero_vec) * adj [n_atom * n_atom]
            => if adjacency element's value bigger than 1e-6(if certain relation exists) then attention value becomes e's element at same location else zero vector's element at same location
        self.gate(x + zero_vec) [n_atom * 1]
        h_prime = relu(attention * h) [n_atom * n_out_feature]
        coeff = Sigmoid(self.gate(x + zero_vec)).repeat(1, 1, X.size(-1)) [n_atom * n_in_feature]
            => working as coefficient indicating importance ratio between X and att_result. Coefficient multiplies same attention values to all elements in single row.
        return coeff * X + (1-coeff) * h_prime
        choose attention component via tr(WXB + tr(WX)) that component at the same place
        in adjacency matrix has bigger value than 1e-6 then apply softmax function to
        attention matrix, multiply adjacency matrix to that then multiply it with WX
        :param x:   atom feature one-hot vector of ligand or protein molecule
        :param adj: adjacency matrix of ligand or protein molecule
        :return:    attention-multiplied matrix
        """
        h = self.W(x)
        e = torch.einsum("ijl,ikl->ijk", (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 * torch.ones_like(e)  # to make softmax result value to zero
        attention = torch.where(adj > 1e-6, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        attention = attention * adj
        h_prime = F.relu(torch.einsum("aij,ajk->aik", (attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(
            1, 1, x.size(-1)
        )
        retval = coeff * x + (1 - coeff) * h_prime
        return retval


class GConv_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GConv_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.gate = nn.Linear(n_out_feature * 2, 1)
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)

    def forward(self, x, adj):
        m = self.W(x)
        m = F.relu(torch.einsum("xjk,xkl->xjl", (adj.clone(), m)))
        feature_size = m.size(-1)
        retval = self.C(m.reshape(-1, feature_size), x.reshape(-1, feature_size))
        retval = retval.reshape(x.size(0), x.size(1), x.size(2))

        # x = torch.bmm(adj, x)
        return retval


class ConcreteDropout(nn.Module):
    def __init__(
        self,
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.1,
    ):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x1, layer, additional_args=None):
        p = torch.sigmoid(self.p_logit)
        if additional_args is None:
            out = layer(self._concrete_dropout(x1, p))
        else:
            out = layer(x1, *aditional_args)

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.0 - p) * torch.log(1.0 - p)

        input_dimensionality = x1[
            0
        ].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (
            torch.log(p + eps)
            - torch.log(1 - p + eps)
            + torch.log(unif_noise + eps)
            - torch.log(1 - unif_noise + eps)
        )

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p


class MultiHeadAttention(nn.Module):
    def __init__(self, args, ninfo):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.ninfo = ninfo
        self.ligand_wq = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.ligand_wk = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.ligand_wv = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.protein_wq = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.protein_wk = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.protein_wv = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.batch = args.batch_size
        if args.ngpu:
            shape = (self.batch // args.ngpu, ninfo, args.dim_gnn)
        elif args.ngpu_on_train:
            shape = (self.batch // args.ngpu_on_train, ninfo, args.dim_gnn)
        else:
            shape = (self.batch, ninfo, args.dim_gnn)
        self.seed_vector = np.ones(shape, dtype=np.float64)
        self.depth = args.dim_gnn // ninfo
        self.q = nn.Parameter(
            torch.from_numpy(self.seed_vector).float(), requires_grad=True
        )
        self.scale = nn.Parameter(
            torch.sqrt(torch.FloatTensor([self.depth])), requires_grad=False
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        1. cut input X into ninfo number of tensors
        2. calculate multiple attentions for each tensors
        3. concat all attention and multiply to all tensors
        """
        batch_size = self.q.shape[0]
        ligand = x.sum(-1).unsqueeze(-1)
        ligand_embedded = ligand.repeat(1, 1, self.args.dim_gnn)
        protein = x.sum(1).unsqueeze(-1)
        protein_embedded = protein.repeat(1, 1, self.args.dim_gnn)

        ligand_q = self.ligand_wq(self.q)
        ligand_k = self.ligand_wk(ligand_embedded)
        ligand_v = self.ligand_wv(ligand_embedded)
        ligand_q = self._split_heads(ligand_q)
        ligand_k = self._split_heads(ligand_k)
        ligand_v = self._split_heads(ligand_v)

        protein_q = self.protein_wq(self.q)
        protein_k = self.protein_wk(protein_embedded)
        protein_v = self.protein_wv(protein_embedded)
        protein_q = self._split_heads(protein_q)
        protein_k = self._split_heads(protein_k)
        protein_v = self._split_heads(protein_v)

        ligand_h = self._multi_head_attention(ligand_q, ligand_k, ligand_v)
        ligand_h = ligand_h.view(batch_size, -1, self.ninfo, self.depth)

        protein_h = self._multi_head_attention(protein_q, protein_k, protein_v)
        protein_h = protein_h.view(batch_size, -1, self.ninfo, self.depth)

        total_h = torch.cat([ligand_h, protein_h], -1)
        total_h = total_h.sum(-1)

        return total_h

    def _split_heads(self, x):
        x = x.view(x.shape[0], -1, self.ninfo, self.depth)
        x = x.permute((0, 2, 1, 3))

        return x

    def _multi_head_attention(self, xq, xk, xv):
        matmul_qk = torch.matmul(xq, torch.transpose(xk, 2, 3))
        attn = matmul_qk / self.scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, xv)

        return out


class NewMultiHeadAttention(nn.Module):
    def __init__(self, args, ninfo):
        super(NewMultiHeadAttention, self).__init__()
        self.args = args
        self.ninfo = ninfo
        self.embedding = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.wq = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.wk = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.wv = nn.Linear(args.dim_gnn, args.dim_gnn)
        self.batch = args.batch_size
        self.seed_vector = np.ones((self.batch, 4, args.dim_gnn), dtype=np.float64)
        self.depth = args.dim_gnn // ninfo
        self.q = nn.Parameter(
            torch.from_numpy(self.seed_vector).float(), requires_grad=True
        )
        self.scale = nn.Parameter(
            torch.sqrt(torch.FloatTensor([self.depth])), requires_grad=False
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        1. cut input X into ninfo number of tensors
        2. calculate multiple attentions for each tensors
        3. concat all attention and multiply to all tensors
        """
        batch_size = self.q.shape[0]
        info = torch.einsum("ijk,ikl->ijl", x, torch.transpose(x, 1, 2))
        info = info.sum(-1).unsqueeze(-1).repeat(1, 1, self.args.dim_gnn)
        info_embedded = self.embedding(info)

        q = self.wq(self.q)
        k = self.wk(info_embedded)
        v = self.wv(info_embedded)

        split_q = self._split_heads(q)
        split_k = self._split_heads(k)
        split_v = self._split_heads(v)

        h = self._multi_head_attention(split_q, split_k, split_v)
        h = h.view(batch_size, -1, self.args.dim_gnn)

        return h

    def _split_heads(self, x):
        x = x.view(x.shape[0], -1, self.ninfo, self.depth)
        x = x.permute((0, 2, 1, 3))
        return x

    def _multi_head_attention(self, xq, xk, xv):
        matmul_qk = torch.matmul(xq, torch.transpose(xk, 2, 3))
        attn = matmul_qk / self.scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, xv)

        return out


class GraphAttention(nn.Module):
    def __init__(self, args, ninfo):
        super(GraphAttention, self).__init__()
        self.args = args
        self.ninfo = ninfo  # 4
        self.wq = [nn.Linear(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        self.wk = [nn.Linear(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        self.wv = [nn.Linear(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x_info = torch.einsum("ijk,ikl->ijl", x, torch.transpose(x, 1, 2))
        # x_info = x.sum(-1)
        embedded = self._arbit_embedding(x_info)
        h_list = []
        for i in range(self.args.n_gnn):
            q = self.wq[i](embedded)
            k = self.wk[i](embedded)
            v = self.wv[i](embedded)
            attn = self._attn_matrix(q, v, x_info)
            h = torch.bmm(attn, v)
            h_list.append(h)
        h = torch.cat(h_list, -1)
        h = self._arbit_embedding(h)
        h = self.relu(h)

        return h

    def _arbit_embedding(self, vec):
        dim = vec.shape[-1]
        embedding = nn.Linear(dim, self.args.dim_gnn)
        embedded = embedding(vec)

        return embedded

    def _attn_matrix(self, q, k, info):
        scale = torch.sqrt(torch.Tensor([k.shape[-1]]))
        attn = torch.einsum("ijk,ikl->ijl", q, torch.transpose(k, 1, 2))
        attn = torch.bmm(attn, info)
        attn /= scale
        attn = self.tanh(attn)

        return attn


class ConvBlock(nn.Module):
    def __init__(
        self, in_feature, out_feature, do=0.0, stride=1, kernel=3, pad=1, bn=True
    ):
        super(ConvBlock, self).__init__()
        self.block = []
        self.block.append(nn.Conv3d(in_feature, out_feature, kernel, stride, pad))
        if bn:
            self.block.append(nn.BatchNorm3d(out_feature))
        self.block.append(nn.ReLU())
        if do != 0:
            self.block.append(nn.Dropout3d(p=do))
        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


class PredictBlock(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, is_last):
        super(PredictBlock, self).__init__()
        self.block = []
        self.block.append(nn.Linear(in_feature, out_feature))
        if not is_last:
            self.block.append(nn.Dropout(p=dropout))
            self.block.append(nn.ReLU())
        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)
