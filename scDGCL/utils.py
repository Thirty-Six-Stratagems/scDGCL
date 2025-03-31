import torch
import numpy as np
import math
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(use_cpu):
    if use_cpu is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    return device



def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs':
            {'nesterov': False,
             'weight_decay': 0.0001,
             'momentum': 0.9,
             },
        'scheduler': 'cosine',
        'scheduler_kwargs': {'lr_decay_rate': 0.1},
    }

    new_lr = None
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)
    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr



def graph_cluster_embedding(embedding, cluster_number, real_label, save_pred=False, cluster_methods=None,
                            use_cosine=False):
    if cluster_methods is None:
        cluster_methods = ["AgglomerativeClustering"]
    result = {"t_clust": time.time()}
    if "AgglomerativeClustering" in cluster_methods:
        if use_cosine:
            distance_matrix = cosine_distances(embedding)
            agglomerative = AgglomerativeClustering(n_clusters=cluster_number, metric='precomputed', linkage='complete')
            pred = agglomerative.fit_predict(distance_matrix)
        else:
            agglomerative = AgglomerativeClustering(n_clusters=cluster_number, metric='euclidean', linkage='ward')
            pred = agglomerative.fit_predict(embedding)

        if real_label is not None:
            result["ari"] = round(adjusted_rand_score(real_label, pred), 4)
            result["nmi"] = round(normalized_mutual_info_score(real_label, pred), 4)
        result["t_agglom"] = time.time() - result["t_clust"]
        if save_pred:
            result["pred"] = pred
    return result
