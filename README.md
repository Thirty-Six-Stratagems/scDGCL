# scDGCL: A Dual-level and Graph-constrained Contrastive Learning Method for Single-cell RNA Sequencing Data Clustering
![model](scDGCL/scDGCL.png)
## Requirements
- python : 3.8.19
- scanpy : 1.8.2
- sklearn : 1.3.2
- torch : 2.0.1
- CUDA : 11.7
- torch-geometric : 2.5.3


## 文件说明

GSM目录下的脚本为GSM预处理模块的具体实现，每个脚本对应的功能如下： 
- Gene_Selection.py : GSM 的主执行脚本；
- SSA.py : GSM 模块的主体；
- Preprocessing.py : 数据预处理脚本（Scanpy流程）；
- Cluster.py : SSA 搜索过程中计算当前解得分的聚类实现脚本；
- trans_csv_to_h5.py : GSM 处理后基因子集完整信息收集转换脚本。  

scDGCL一级目录下的脚本对应的功能如下：  
- main.py : scDGCL 的主执行脚本；
- scDGCL.py : scDGCL 模型的具体架构实现；
- GGM.py : 图构建脚本；
- train.py : scDGCL 的训练流程；
- utils.py : 为实现 scDGCL 的辅助函数。

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
