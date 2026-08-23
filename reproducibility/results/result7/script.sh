Rscript preprocess/combine_subsets.R --task covid
Rscript preprocess/combine_subsets.R --task flu_a
Rscript preprocess/combine_subsets.R --task tuber
python preprocess/split_mat.py --task covid
python preprocess/split_mat.py --task flu_a
python preprocess/split_mat.py --task tuber
python train.py --task atlas_disease_cl --cuda 0 --REPLAY 1 --denovo 0 --use_shm 1 --start 1