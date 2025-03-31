import pandas as pd
import scanpy as sc
import h5py
import os
import numpy as np
def read_data(data_path):
    X = pd.read_csv(data_path, header=0, index_col=0, sep=',')
    adata = sc.AnnData(X)
    X = adata.X
    return X
# dataset_names = ["Pollen", "Klein", "PBMC", "Quake_10x_Bladder", 'Quake_Smart-seq2_Trachea',
#                 "Muraro", 'Quake_Smart-seq2_Heart', "Quake_Smart-seq2_Lung", "Romanov", "Quake_10x_Spleen"]
dataset_names = ["Pollen"]
for datasetname in dataset_names:
    data_path = f"Gene_selection_data/{datasetname}/subset.csv"
    with h5py.File(f"../H5_data/{datasetname}.h5", 'r') as data_h5:
        Y = np.array(data_h5.get('Y')).reshape(-1)
    output_dir = '../Selected_data'
    filename = f"{datasetname}.h5"
    X= read_data(data_path)
    output_path = os.path.join(output_dir, filename)
    with h5py.File(output_path, 'w') as outfile:
        outfile.create_dataset('X', data=X, dtype='float64')
        outfile.create_dataset('Y', data=Y, dtype='int32')
        print(f"Processed and saved: {filename}")
