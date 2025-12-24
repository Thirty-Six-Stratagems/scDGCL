import os
from SSA import SSA
from Preprocessing import read_data


def run_multiple_datasets(dataset_names, input_dir, get_HVGs=True, k=None, max_iter=60,
                          save_path="Gene_selection_data"):
    for i, dataset_name in enumerate(dataset_names):
        print(f"\n=================== Processing Dataset: {dataset_name} ===================")

        data_path = os.path.join(input_dir, f"{dataset_name}.h5")

        dataset_save_path = os.path.join(save_path, dataset_name)
        os.makedirs(dataset_save_path, exist_ok=True)
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found for {dataset_name}. Skipping.")
            continue
        data, label, k_value = read_data(data_path, get_HVGs, k)
        print(f"=================== Run Gene_Selection for Dataset: {dataset_name} ===================")
        ssa = SSA(data=data, max_iter=max_iter, data_label=label, k=k_value, RESULT_PATH=dataset_save_path)
        subset = ssa.run_task()
        print(f"Results for dataset {dataset_name} saved in {dataset_save_path}")
    print("\nAll datasets processed successfully.")


if __name__ == '__main__':
    input_dir = "../H5_data"
    dataset_names = ["Pollen", "Klein", "PBMC", "Quake_10x_Bladder", 'Quake_Smart-seq2_Trachea',
                "Muraro", 'Quake_Smart-seq2_Heart', "Quake_Smart-seq2_Lung", "Romanov", "Quake_10x_Spleen"]
    run_multiple_datasets(
        dataset_names=dataset_names,
        input_dir=input_dir,
        get_HVGs=True,
        k=None,
        max_iter=60,
        save_path="Gene_selection_data"
    )
