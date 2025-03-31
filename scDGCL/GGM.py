import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from scipy.spatial import distance
from tdigest import TDigest



def chunked_quantile(tensor, q, chunk_size=1000):
    chunks = tensor.split(chunk_size)
    quantiles = [chunk.quantile(q) for chunk in chunks]
    return torch.tensor(quantiles).mean()


def normalize_data(x):
    mean = x.mean(1).unsqueeze(1)
    std = x.std(1).unsqueeze(1)
    return (x - mean) / (std + 1e-8)


def pad_to_nearest_multiple(x, multiple):
    _, current_dim = x.shape
    target_dim = math.ceil(current_dim / multiple) * multiple
    padding_size = target_dim - current_dim

    if padding_size > 0:
        padding = torch.zeros(x.shape[0], padding_size, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=1)
    else:
        x_padded = x
    return x_padded, padding_size

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, h, dropout=0):
        super(GraphConstructor, self).__init__()
        self.h = h
        self.d_k = math.ceil(input_dim / h)
        self.padded_dim = self.d_k * h
        self.linears = clones(nn.Linear(self.padded_dim, self.padded_dim), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)

    def forward(self, query, key, attn_rate):
        query_padded, _ = pad_to_nearest_multiple(query, self.h)
        key_padded, _ = pad_to_nearest_multiple(key, self.h)

        query, key = [l(x).view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query_padded, key_padded))]

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        threshold = chunked_quantile(attns, attn_rate)
        adj = torch.where(attns >= threshold, attns, torch.zeros_like(attns))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def construct_att_graph(x, h, attn_rate, dropout=0):
    x = normalize_data(x)

    device = x.device
    x_padded, padding_size = pad_to_nearest_multiple(x, h)
    graph_constructor = GraphConstructor(x_padded.shape[1], h, dropout).to(device)
    adj = graph_constructor(x_padded, x_padded, attn_rate)
    adj = adj - torch.diag_embed(torch.diag(adj))
    with torch.no_grad():
        adj_cpu = adj.cpu().numpy()
        gcn_adj = adj_cpu.copy()
        non_zero_mask = adj_cpu != 0
        if non_zero_mask.any():
            min_val = adj_cpu[non_zero_mask].min()
            max_val = adj_cpu[non_zero_mask].max()
            adj_cpu[non_zero_mask] = (adj_cpu[non_zero_mask] - min_val) / (max_val - min_val)

        adj_sparse = csr_matrix(adj_cpu)
        edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

    return edge_index, edge_weight, gcn_adj


def generateAdj(gene_exp, k=10, th_rate=10):
    featureMatrix = gene_exp
    if isinstance(featureMatrix, torch.Tensor):
        featureMatrix = featureMatrix.cpu().numpy()
    else:
        featureMatrix = np.array(featureMatrix)
    n = featureMatrix.shape[0]
    adjMatrix = np.zeros((n, n))
    distMat = distance.cdist(featureMatrix, featureMatrix, 'cityblock')
    distMat_tensor = torch.from_numpy(distMat)

    threshold = np.percentile(distMat, th_rate)

    for i in range(n):
        distances, indices = torch.topk(distMat_tensor[i], k=k + 1, largest=False)
        for j, dist in zip(indices[1:], distances[1:]):
            if dist <= threshold:
                weight = 1 - (dist / threshold)
                weight = max(0, min(1, weight.item()))
                adjMatrix[i, j] = weight
                adjMatrix[j, i] = weight

        if np.sum(adjMatrix[i]) == 0:
            adjMatrix[i, i] = 1

    adj_sparse = csr_matrix(adjMatrix)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)

    return adjMatrix, edge_index, edge_weight


def combine_graphs(adj1, adj2):
    if isinstance(adj1, torch.Tensor):
        adj1 = adj1.cpu().numpy()
    if isinstance(adj2, torch.Tensor):
        adj2 = adj2.cpu().numpy()

    assert adj1.shape == adj2.shape, "Adjacency matrices must have the same shape"

    combined_adj = np.zeros_like(adj1)
    mask = (adj1 != 0) & (adj2 != 0)
    combined_adj[mask] = adj1[mask] + adj2[mask]
    nonzero_count = np.count_nonzero(combined_adj)
    if nonzero_count < 100:
        for i in range(combined_adj.shape[0]):
            if np.sum(combined_adj[i]) == 0:
                combined_adj[i, i] = 1
    combined_adj_sparse = csr_matrix(combined_adj)
    edge_index, edge_weight = from_scipy_sparse_matrix(combined_adj_sparse)

    return edge_index, edge_weight, combined_adj
