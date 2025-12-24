import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import h5py
def read_data(data_path, get_HVGs=True, k=None):
    with h5py.File(data_path, 'r') as data_h5:
        data = np.array(data_h5.get('X'))
        print("data shape", data.shape)
        real_label = np.array(data_h5.get('Y')).reshape(-1)
    print("data shape = {}".format(data.shape))
    if get_HVGs == True:
        data, _ = Selecting_highly_variable_genes(data, 2000)
    else:
        data = np.array(data).T
    label = None
    if real_label is not None:
        label = real_label
        k = len(np.unique(label))
        print("label is not none. label shape = {}, k = {}".format(label.shape, k))
    else:
        if k is None:
            print("Evaluate the number of K...")
            k = Evaluate_k(data)
        print("label is none. k = {}".format(k))

    return data, label, k


def Selecting_highly_variable_genes(X, highly_genes=2000):
    adata = sc.AnnData(X)
    if isinstance(adata, sc.AnnData):
        adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    adata.X = adata.X.astype(np.float32)
    if adata.X.size < 50e6:
        if sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata)
    data = adata.X
    return data, adata


def Evaluate_k(data):
    adata = sc.AnnData(data)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.louvain(adata)
    k = len(np.unique(adata.obs['louvain']))
    return k
