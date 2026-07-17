MIRACLE is an online learning framework designed for scalable multimodal integration. 

# User-friendly Package

This repository is dedicated to reproducing results 2–7 from the manuscript. For a user-friendly package, please visit [sc-miracle/miracle](https://github.com/sc-miracle/miracle).

# Guidance for Reproducing the Manuscript Results

## Preparation

This package is implemented in Python, with preprocessing primarily handled in R. To install the required Python packages using Conda, run:

```bash
conda create -n miracle python=3.11.11
conda activate miracle
pip install -r requirements.txt
```

### **Dependencies**

#### **Python Packages:**

| Package      | Version  |
| ------------ | -------- |
| Python       | v3.11.11 |
| PyTorch      | v2.1.2   |
| scikit-learn | v1.3.2   |
| Scanpy       | v1.9.6   |
| scib         | v1.1.4   |

#### **R Packages:**

| Package | Version                              | Notes                               |
| ------- | ------------------------------------ | ----------------------------------- |
| Seurat  | v4.1.0                               | For data preprocessing and analysis |
| Seurat  | [v5.2.1](https://satijalab.org/seurat/) | For sketch sampling                 |
| Signac  | v1.6.0                               |                                     |

---

## **Result 2: Continual Integration Using Limited Memory**

### **Step 1: Data Preparation**

For quality control, refer to `reproducibility-code/result2-DHCM/preprocess_dcm_hcm.ipynb`. This experiment generates 42 batches of data, all sharing the same features.

```bash
Rscript preprocess/combine_subsets.R --task dcm_hcm
python preprocess/split_mat.py --task dcm_hcm
for i in {0..41}
do
mkdir ./data/processed/dcm_hcm_$[i+1]
mkdir ./data/processed/dcm_hcm_$[i+1]/subset_0
ln -sr ./data/processed/dcm_hcm/subset_$i/* ./data/processed/dcm_hcm_$[i+1]/subset_0/
ln -sr ./data/processed/dcm_hcm/feat ./data/processed/dcm_hcm_$[i+1]/
done
```

### **Step 2: Integration**

```bash
python train.py --cuda 0 --task dcm_hcm --exp_prefix continual_ --max_size 200000 --actions train predict_all_latent subsample --epoch_list 500
```

### **Step 3: Evaluation and Visualization**

```bash
python eval/benchmark_batch_bio.py --task dcm_hcm --experiment continual_41
```

Since computing the UMAP for variable **u** is time-consuming, we visualize only variable **c** by setting `use_u` to 0.

```bash
Rscript comparison/midas_embed.r --task dcm_hcm --experiment continual_41 --use_u 0
```

---

## **Result 3: Continual Integration Across Batches and Cell Types**

### **Step 1: Data Preparation**

```bash
Rscript preprocess/combine_subsets.R --task p1_0 && py preprocess/split_mat.py --task p1_0 & 
Rscript preprocess/combine_subsets.R --task p2_0 && py preprocess/split_mat.py --task p2_0 & 
Rscript preprocess/combine_subsets.R --task p3_0 && py preprocess/split_mat.py --task p3_0 & 
Rscript preprocess/combine_subsets.R --task p4_0 && py preprocess/split_mat.py --task p4_0 & 
Rscript preprocess/combine_subsets.R --task p5_0 && py preprocess/split_mat.py --task p5_0 & 
Rscript preprocess/combine_subsets.R --task p6_0 && py preprocess/split_mat.py --task p6_0 & 
Rscript preprocess/combine_subsets.R --task p7_0 && py preprocess/split_mat.py --task p7_0 & 
Rscript preprocess/combine_subsets.R --task p8_0 && py preprocess/split_mat.py --task p8_0
```

### **Step 2: Integration**

```bash
python train.py --cuda 0 --task teadog --exp_prefix continual_ --actions train predict_all_latent subsample
```

### **Step 3: Evaluation and Visualization**

```bash
python eval/benchmark_batch_bio.py --task teadog --experiment continual_7
```

```bash
Rscript comparison/midas_embed.r --task teadog --experiment continual_7
```

---

## **Result 4: Continual Mosaic Integration**

### **Step 1: Data Preparation**

```bash
Rscript preprocess/combine_subsets.R --task lll_ctrl && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task lll_ctrl && py preprocess/split_mat.py --task lll_ctrl &
Rscript preprocess/combine_subsets.R --task lll_stim && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task lll_stim && py preprocess/split_mat.py --task lll_stim &
Rscript preprocess/combine_subsets.R --task dig_ctrl && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task dig_ctrl && py preprocess/split_mat.py --task dig_ctrl &
Rscript preprocess/combine_subsets.R --task dig_stim && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task dig_stim && py preprocess/split_mat.py --task dig_stim &
Rscript preprocess/combine_subsets.R --task W3 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W3 && py preprocess/split_mat.py --task W3 &
Rscript preprocess/combine_subsets.R --task W4 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W4 && py preprocess/split_mat.py --task W4 &
Rscript preprocess/combine_subsets.R --task W5 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W5 && py preprocess/split_mat.py --task W5 &
Rscript preprocess/combine_subsets.R --task W6 && Rscript preprocess/combine_unseen.R --reference teadog_label_mask --task W6 && py preprocess/split_mat.py --task W6
```

### **Step 2: Integration**

```bash
python train.py --cuda 0 --task teadog --exp_prefix continual_ --actions train predict_all_latent subsample
```

### **Step 3: Evaluation and Visualization**

```bash
python eval/benchmark_batch_bio.py --task teadog --experiment continual_7
```

```bash
Rscript comparison/midas_embed.r --task teadog --experiment continual_7
```

## **Result 5: Continual Construction of a Cross-Tissue Multimodal Atlas**

### **Step 1: Offline Integration of Reference Atlas**

```bash
Rscript preprocess/combine_subsets.R --task atlas_new_no_neap
py preprocess/split_mat.py --task atlas_new_no_neap
CUDA_VISIBLE_DEVICES=0 python run.py --task atlas_new_no_neap --experiment offline --actions train predict_all_latent --use_shm 1

Rscript preprocess/combine_subsets.R --task atlas_new
py preprocess/split_mat.py --task atlas_new
CUDA_VISIBLE_DEVICES=1 python run.py --task atlas_new --experiment offline --actions train predict_all_latent --use_shm 1

Rscript preprocess/combine_subsets.R --task atlas_tissues
py preprocess/split_mat.py --task atlas_tissues
CUDA_VISIBLE_DEVICES=2 python run.py --task atlas_tissues --experiment offline --actions train predict_all_latent --use_shm 1
```

### **Step 2: Continual Integration of PBMC Query Datasets**

**Data Preparation**

```bash
Rscript preprocess/combine_subsets.R --task query_neat
Rscript preprocess/combine_subsets.R --task asap
Rscript preprocess/combine_subsets.R --task asap_cite
Rscript preprocess/combine_unseen.R --reference atlas_new_no_neap --task query_neat
Rscript preprocess/combine_unseen.R --reference atlas_new_no_neap --task asap
py preprocess/split_mat.py --task query_neat
py preprocess/split_mat.py --task asap
py preprocess/split_mat.py --task asap_cite
```

**Integration**

```bash
python train.py --cuda 3 --task new_query_cl --actions train predict_subsample subsample predict_all --denovo 0
```

### **Step 3: Continual Integration of Cross-Tissue Query Datasets**

**Data Preparation**

```bash
Rscript preprocess/combine_subsets.R --task tonsil
Rscript preprocess/combine_subsets.R --task bone_marrow_02
Rscript preprocess/combine_subsets.R --task spleen
Rscript preprocess/combine_unseen.R --reference atlas_new --task bone_marrow_02
py preprocess/split_mat.py --task tonsil
py preprocess/split_mat.py --task bone_marrow_02
py preprocess/split_mat.py --task spleen
```

**Integration**

```bash
python train.py --cuda 4 --task atlas_tissues_cl --actions train predict_subsample subsample predict_all --denovo 0
```

---

## **Result 6: Label Transfer for Cross-Tissue Mosaic Data**

### **Step 1: Label Transfer of PBMC Query Datasets**

```bash
python train.py --cuda 0 --task single_query_neat_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 1 --task single_asap_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 2 --task single_asap_cite_cl --actions train predict_subsample subsample predict_all --denovo 0
```

### **Step 2: Label Transfer of Cross-Tissue Query Datasets**

```bash
python train.py --cuda 3 --task single_tonsil_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 4 --task single_bm_cl --actions train predict_subsample subsample predict_all --denovo 0
python train.py --cuda 5 --task single_spleen_cl --actions train predict_subsample subsample predict_all --denovo 0
```

## **Result 7: Continual Integration and Analysis of Diverse Respiratory Infection Data**

```bash
python train.py --task atlas_disease_cl --cuda 0 --REPLAY 1 --denovo 0 --use_shm 1 --start 1
```

---

## **Analysis and Comparison Results**

For further details, refer to `reproducibility-code`.
