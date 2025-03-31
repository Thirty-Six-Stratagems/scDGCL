import h5py
import torch
import numpy as np
import train
import csv

Tensor = torch.cuda.FloatTensor


def process_dataset(dataset_name, h5_datasets):
    final_data_list = []
    if dataset_name in h5_datasets:
        with h5py.File(f"Selected_data/{dataset_name}.h5", 'r') as data_h5:
            gene_exp = np.array(data_h5.get('X'))
            real_label = np.array(data_h5.get('Y')).reshape(-1)
    print(f"The gene expression matrix shape for {dataset_name} is {gene_exp.shape}...")
    cluster_number = np.unique(real_label).shape[0]
    print(f"The real clustering num for {dataset_name} is {cluster_number}...")

    for i in range(10):
        final_results = train.run(
            gene_exp=gene_exp,
            cluster_number=cluster_number,
            dataset=dataset_name,
            real_label=real_label,
            epochs=200, lr=0.2,
            temperature=0.07,
            dropout=0.9,
            save_pred=True,
            cluster_methods="KMeans",
            batch_size=200,
            m=0.5, noise=0.1,
            th_rate=10,
            attn_rate=0.95,
            prob_feature=0.1,
            prob_edge=0.5)

        print(f"final_Results for {dataset_name}:")
        print("ARI:    " + str(final_results["ari"]))
        print("NMI:    " + str(final_results["nmi"]))
        final_cluster_labels = final_results.get("pred", None)
        final_embedding = final_results.get("features", None)
        final_data_list.append([float(final_results["ari"]), float(final_results["nmi"])])

    return final_data_list


if __name__ == "__main__":
    datasets = ["Pollen", "Klein", "PBMC", "Quake_10x_Bladder", 'Quake_Smart-seq2_Trachea',
                "Muraro", 'Quake_Smart-seq2_Heart', "Quake_Smart-seq2_Lung", "Romanov", "Quake_10x_Spleen"]
    final_datas_list = []
    for dataset_name in datasets:
        final_data_list = process_dataset(
            dataset_name,
            datasets)
        final_datas_list.append(final_data_list)
        with open('ari_nmi_data.csv', mode='a', encoding='utf-8', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([dataset_name])
            csv_write.writerow(final_data_list)