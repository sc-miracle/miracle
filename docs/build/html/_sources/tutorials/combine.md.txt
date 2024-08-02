# Combine batches and generate inputs

 We assume that these batches are observed one by one. Therefore, we do not really combine their features. We use the following command to fetch their expression matrix.

```bash
Rscript preprocess/combine_subsets.R --task lll_ctrl && Rscript preprocess/combine_unseen.R --reference atlas --task lll_ctrl && py preprocess/split_mat.py --task lll_ctrl &&
Rscript preprocess/combine_subsets.R --task lll_stim && Rscript preprocess/combine_unseen.R --reference atlas --task lll_stim && py preprocess/split_mat.py --task lll_stim &&
Rscript preprocess/combine_subsets.R --task dig_ctrl && Rscript preprocess/combine_unseen.R --reference atlas --task dig_ctrl && py preprocess/split_mat.py --task dig_ctrl &&
Rscript preprocess/combine_subsets.R --task dig_stim && Rscript preprocess/combine_unseen.R --reference atlas --task dig_stim && py preprocess/split_mat.py --task dig_stim
```
