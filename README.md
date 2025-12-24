# scDGCL: A Dual-level and Graph-constrained Contrastive Learning Method for Single-cell RNA Sequencing Data Clustering
![model](scDGCL/scDGCL.png)
## Requirements
- python : 3.8.19
- scanpy : 1.8.2
- sklearn : 1.3.2
- torch : 2.0.1
- CUDA : 11.7
- torch-geometric : 2.5.3

## Datasets
- Pollen (SRP041736) : https://www.ncbi.nlm.nih.gov/sra
- Trachea (GSE109774) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Lung (GSE109774) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Bladder (GSE109774) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Heart (GSE109774) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Spleen (GSE109774) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774
- Muraro (GSE85241) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241
- Klein (GSE65525) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525
- Romanov (GSE74672) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74672
- PBMC (SRP073767) : https://www.10xgenomics.com/datasets/4-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0
- Chen (GSE87544) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544
- Macosko (GSE63473) : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63473
- Baron1, Baron2, Baron3 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133


## Files Description

The scripts in the GSM directory are the specific implementations of the GSM preprocessing module. Each script's function is as follows:   
- Gene_Selection.py : The main execution script for GSM;
- SSA.py : The main body of the GSM module;
- Preprocessing.py : Data preprocessing script (Scanpy workflow);
- Cluster.py : The clustering implementation script for calculating the current solution score during the SSA search process;
- trans_csv_to_h5.py : The script for collecting and converting complete gene subset information after GSM processing.

The scripts in the scDGCL directory have the following functions:
- main.py : The main execution script for scDGCL;
- scDGCL.py : The specific architecture implementation of the scDGCL model;
- GGM.py : Graph construction script;
- train.py : The training process for scDGCL;
- utils.py : Auxiliary functions for implementing scDGCL.

## Usage  

To use GSM for data processing, switch to the GSM directory and run the following code:
```bash
cd scDGCL/GSM
python Gene_Selection.py
```
After running the script, we obtain the subset of gene expression data selected from the initial H5 data and save it in CSV format. Then, run the trans_csv_to_h5.py script in the GSM directory to convert the CSV data to .h5 format, attaching labels corresponding to each cell:
```bash
cd scDGCL/GSM
python trans_csv_to_h5.py
```
In this way, we obtain fully preprocessed data stored in .h5 file format in the Selected_data directory.

Once the data is processed, we can execute the scDGCL method for clustering using the following command:
```bash
cd scDGCL
python main.py
```
Besides, we provide a pre-processed Pollen Muraro dataset. You can skip the GSM processing steps and directly execute the following command to use scDGCL:
```bash
cd scDGCL
python main.py
```
