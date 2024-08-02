Data Preprocessing
==================

## Quality control and feature selection

The count matrices of RNA and ADT were processed via `Seurat`. The ATAC fragment files were processed using `Signac`, and peaks were called via the Python package `MACS2`. We performed quality control separately for each batch. Briefly, metrics of detected **gene number per cell, total UMI number, percentage of mitochondrial RNA reads, total protein tag number, total fragment number, transcription start site score and nucleosome signal** were evaluated. We manually checked the distributions of these metrics and set customized criteria to filter low-quality cells in each batch. For each batch, we adopted common normalization strategies for RNA, ADT and ATAC data, respectively. Specifically, for RNA data, UMI count matrices are normalized and log transformed using the `NormalizeData` function in Seurat. For ADT data, tag count matrices are centered log ratio normalized using the `NormalizeData` function in Seurat. For ATAC data, fragment matrices are term frequency inverse document frequency normalized using the `RunTFIDF` function in Signac. To integrate batches profiled by various technologies, we need to create a union of features for RNA, ADT and ATAC data, respectively. For RNA data, first, low-frequency genes are removed based on gene occurrence frequency across all batches. We then select **4,000** highly variable genes using the `FindVariableFeatures` function with default parameters in each batch. The union of these highly variable genes is ranked using the `SelectIntegrationFeatures` function, and the top 4,000 genes are selected. In addition, we also retain genes that encode proteins targeted by the antibodies. For ADT data, the union of antibodies in all batches is retained for data integration. For ATAC data, we used the `reduce` function in Signac to merge all intersecting peaks across batches and then recalculated the fragment counts in the merged peaks. The merged peaks are used for data integration. The input data are UMI counts for RNA data, tag counts for ADT data and binarized fragment counts for ATAC data. For each modality, the union of features from all batches are used. Counts of missing features are set to 0. Binary feature masks are generated accordingly, where 1 and 0 denote presented and missing features, respectively.

## Combine batches

We provide an R script that combines different batches and obtains their union features. The expression matrix will be extracted from the Seurat object and saved as CSV files. To run combine the batches of data using our codes, a standard data configure file is necessary. Please add your item in the file `./configs/data.toml`.

|      key      | explanation                                                                                                                                                                         |
| :------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| raw_data_dirs | The directory corresponding to each batch. The Seurat objects for RNA, ADT modalities should be<br />stored in the `seurat` folder under the respective batch directory.          |
| raw_data_frags | The location of the corresponding fragment files for ATAC should be under `raw_data_dirs/raw_data_frags`. If the batch <br />does not contain ATAC data, please fill it with "". |
|     combs     | The omics combination corresponding to each batch. Various omics combinations can be achieved by controlling this item.                                                             |
|  comb_ratios  | The proportion of data used for each batch, where 1 indicates that all the data in the dataset will be used.                                                                       |
|    s_joint    | ID for batch.                                                                                                                                                                       |
|  train_ratio  | The proportion of training data, it is recommended to be 1.                                                                                                                         |
|       N       | Mini-batch size. Default is 256.                                                                                                                                                    |

Then run:

```bash
Rscript preprocess/combine_subsets.R --task <task name>
```

To align the ATAC data, use:

```bash
preprocess/combine_unseen.R --reference <reference task> --task <task>
```

## Input Format

MIRACLE does not use the AnnData object as input, but instead divides the data into separate CSV files per cell. This helps reduce the amount of memory consumed during the training process. You can use the following script to split a CSV file containing multiple cell data into multiple CSV files divided by cells.

```bash
python preprocess/split_mat.py --task <task_name>
```

Then you will get a directory containing lots of CSV files like this:

```
#./data/processed/
|-- dataset_1
|   |-- feat  # feature information
|   |   |-- feat_dims.csv
|   |   |-- feat_names_adt.csv
|   |   |-- feat_names_atac.csv
|   |   |-- feat_names_rna.csv
|   |-- subset_0 
|   |   |-- mask # binary mask for rna and adt
|   |   |   |-- rna.csv
|   |   |   |-- adt.csv
|   |   |-- vec  # csv file for every modality of every cell
|   |   |   |-- rna
|   |   |   |   |-- 00000.csv
|   |   |   |   |-- 00001.csv
|   |   |   |   |-- ......
|   |   |   |-- adt
|   |   |   |   |-- 00000.csv
|   |   |   |   |-- 00001.csv
|   |   |   |   |-- ......
|   |   |   |-- atac
|   |   |   |   |-- 00000.csv
|   |   |   |   |-- 00001.csv
|   |   |   |   |-- ......
|   |   |--cell_names.csv
|   |   |--cell_names_sampled.csv # if there is
|   |-- subset_1
|   |-- ......

```

See the tutorials for examples.
