import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from utils import get_device
import math
import torch

device = get_device(use_cpu=None)


class DataAug(nn.Module):
    def __init__(self, dropout=None):
        super(DataAug, self).__init__()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        aug_data = self.drop(x)

        return aug_data


class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.dims = dims
        self.n_stacks = len(self.dims)
        enc = [nn.Linear(self.dims[0], self.dims[1]), nn.BatchNorm1d(self.dims[1]), nn.ReLU(),
               nn.Linear(self.dims[1], self.dims[2]), nn.BatchNorm1d(self.dims[2]), nn.ReLU(),
               nn.Linear(self.dims[2], self.dims[3]), nn.BatchNorm1d(self.dims[3]), nn.ReLU()]

        self.encoder = nn.Sequential(*enc)
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        latent_out = self.encoder(x)
        latent_out = F.normalize(latent_out, dim=1)
        return latent_out


class Projector(nn.Module):
    def __init__(self, contrastive_dim, projector_dim, dim):
        super(Projector, self).__init__()
        self.contrastive_dim = contrastive_dim
        self.dim = dim
        self.projector_dim = projector_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.contrastive_dim, self.projector_dim),
            nn.ReLU(),
            nn.Linear(self.projector_dim, self.dim),
        )

    def forward(self, x):
        latent_out = self.encoder(x)
        return latent_out


class DCL(nn.Module):

    def __init__(self, encoder_1, encoder_2, instance_projector, cluster_projector, class_num,
                 m=0.2):
        super(DCL, self).__init__()

        self.cluster_num = class_num
        self.m = m
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2
        for param_1, param_2 in zip(self.encoder_1.parameters(), self.encoder_2.parameters()):
            param_2.data.copy_(param_1.data)
            param_2.requires_grad = False
        self.instance_projector = instance_projector
        self.cluster_projector = nn.Sequential(
            cluster_projector,
            nn.Softmax(dim=1)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_1, param_2 in zip(self.encoder_1.parameters(), self.encoder_2.parameters()):
            param_2.data = param_2.data * self.m + param_1.data * (1. - self.m)

    def forward(self, cell_1, cell_2):
        E_1 = self.encoder_1(cell_1)
        cell_level_1 = F.normalize(self.instance_projector(E_1), dim=1)
        cluster_level_1 = self.cluster_projector(E_1)
        if cell_2 is None:
            return cell_level_1, cluster_level_1, None, None
        with torch.no_grad():
            self._momentum_update_key_encoder()
            E_2 = self.encoder_2(cell_2)
            cell_level_2 = F.normalize(self.instance_projector(E_2), dim=1)
            cluster_level_2 = self.cluster_projector(E_2)

        return cell_level_1, cluster_level_1, cell_level_2, cluster_level_2


def Graph_Aug(x, adj, prob_feature, prob_edge):
    batch_size, input_dim = x.shape
    feature_mask = torch.bernoulli(torch.full((batch_size, input_dim), 1 - prob_feature)).to(device)
    x_aug = x * feature_mask
    edge_mask = torch.bernoulli(torch.full(adj.shape, 1 - prob_edge)).to(device)
    adj_aug = adj * edge_mask
    return x_aug, adj_aug


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def con_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj > 0, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)


def final_con_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * con_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * con_loss(z_aug, z, adj_aug, tau, hidden_norm)
    return loss


class GCCL(nn.Module):
    def __init__(self, input_dim, gcn_dim, projector_dim, prob_feature, prob_edge, tau, alpha, beta, dropout):
        super(GCCL, self).__init__()
        self.prob_feature = prob_feature
        self.prob_edge = prob_edge
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gcn_1 = GCNConv(input_dim, gcn_dim)
        self.gcn_2 = GCNConv(gcn_dim, projector_dim)
        self.w_imp = nn.Linear(projector_dim, input_dim)
        self.projector = nn.Linear(projector_dim, projector_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = self.dropout(x)
        edge_index = torch.nonzero(adj).T
        edge_weight = adj[edge_index[0], edge_index[1]]

        x_aug, adj_aug = Graph_Aug(x, adj, self.prob_feature, self.prob_edge)
        edge_index_aug = torch.nonzero(adj_aug).T
        edge_weight_aug = adj_aug[edge_index_aug[0], edge_index_aug[1]]

        z = self.gcn_1(x, edge_index, edge_weight=edge_weight)
        z = self.gcn_2(z, edge_index, edge_weight=edge_weight)
        z_aug = self.gcn_1(x_aug, edge_index_aug, edge_weight=edge_weight_aug)
        z_aug = self.gcn_2(z_aug, edge_index_aug, edge_weight=edge_weight_aug)

        x_imp = self.w_imp(z)

        z_projector = self.projector(z)
        z_projector_aug = self.projector(z_aug)
        loss_cl = final_con_loss(self.alpha, self.beta, z_projector, z_projector_aug, adj, adj_aug, self.tau,
                                 hidden_norm=True)

        return z, x_imp, loss_cl


EPS = 1e-8


class Cellconloss(nn.Module):
    def __init__(self, temperature):
        super(Cellconloss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, features):
        device = get_device(use_cpu=None)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        self.batch_size = features.shape[0]
        self.mask = self.mask_correlated_samples(self.batch_size).to(device)
        N = 2 * self.batch_size
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        sim = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        cell_loss = self.criterion(logits, labels)
        cell_loss /= N
        return cell_loss


class Clusterconloss(nn.Module):
    def __init__(self, class_num, temperature):
        super(Clusterconloss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, features_cluster):
        c_i, c_j = torch.unbind(features_cluster, dim=1)
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j
        N = 2 * self.class_num
        c = features_cluster
        cluster_feature = torch.cat(torch.unbind(c, dim=1), dim=0)
        sim = torch.matmul(cluster_feature, cluster_feature.T) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        cluster_loss = loss + ne_loss
        return cluster_loss
