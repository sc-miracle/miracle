# %%
import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py

import scanpy as sc
import tensorflow as tf
import tensorflow_probability as tfp

physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
# data_path = 'TEADOG_paired_full'

# %%
import argparse
parser = argparse.ArgumentParser()
## Task
parser.add_argument('--task', type=str, default='wnn',
    help="Choose a task")
parser.add_argument('--experiment', type=str, default='transfer',
    help="")
parser.add_argument('--ref_id', type=int, default=None,
    help="reference task id")
parser.add_argument('--pretrained', type=str, default=None,
    help="pretrained model path")
parser.add_argument('--actions', type=str, default=['train'], nargs='+',
    help="")
o = parser.parse_args()
task = o.task
# data_root = '/dev/shm/processed/'+task
data_root = 'path/data/processed/'+task

print(o.experiment)
# %%


# %%
import os

mods = []
data = []
n_cells = 0
n_cells_list = [0]
masks_ = []
cell_types = []
batches = []
if o.ref_id is None:
    a = range(8)
else:
    a = range(1)
for i in a:
    if o.ref_id is not None:
        j = 0
        i  = o.ref_id
    else:
        j = i
    m = []
    d = {}
    p = data_root+'/subset_'+str(i)
    print('loading ', p)
    mask = {}
    for f in os.listdir(p+'/mat'):
        print(f)
        mm = f.strip('.csv')
        m.append(mm)
        d[mm]=(pd.read_csv(p+'/mat/'+f, sep=',', index_col=0).to_numpy())
        if mm!='ata':
            mask[mm] = (pd.read_csv(p+'/mask/'+f, sep=',', index_col=0).to_numpy())
    # print(m)
    mods.append(m)
    data.append(d)
    n_cells += len(data[j][mm])
    n_cells_list.append(n_cells_list[-1] + len(data[j][mm]))
    masks_.append(mask)
    batches.append([j for _ in range(len(data[j][mm]))])
    # cell_types.append(pd.read_csv(labels[j]).to_numpy())
    # if 'tea' in task:
    #     cell_types.append(pd.read_csv(teadog_labels[i]).to_numpy())
    # else:
    #     cell_types.append(pd.read_csv(dogma_labels[i]).to_numpy())
batches = np.concatenate(batches)
# cell_types = np.concatenate(cell_types)[:, 1]

# %%
gene_names = pd.read_csv(data_root+'/feat/feat_names_rna.csv',header=0, index_col=0)
# peak_names = pd.read_csv(data_root+'/feat/feat_names_atac.csv',header=0, index_col=0)
ADT_names = pd.read_csv(data_root+'/feat/feat_names_adt.csv',header=0, index_col=0)

# %%
# chunk_atac = np.array([
#     np.sum(np.char.startswith(list(peak_names['x']), 'chr%d-'%i)) for i in range(1,23)
#     ], dtype=np.int32)

# %%

# dim_input_arr = np.array([len(gene_names),len(ADT_names),len(peak_names)])
# print(dim_input_arr)

# %%
X = np.zeros((n_cells, len(gene_names)))
Y = np.zeros((n_cells, len(ADT_names)))
# Z = np.zeros((n_cells, len(peak_names)))
for i, mm in enumerate(mods):
    for m in mm:
        if m=='rna':
            X[n_cells_list[i]:n_cells_list[i+1], :] = data[i][m]
        if m=='adt':
            Y[n_cells_list[i]:n_cells_list[i+1], :] = data[i][m]
        # if m=='ata':
        #     Z[n_cells_list[i]:n_cells_list[i+1], :] = data[i][m]

del data
# %%
# Count peaks in each chromosome (assuming they are ordered)
# chunk_atac = np.array([
#     np.sum(np.char.startswith(list(peak_names['x']), 'chr%d-'%i)) for i in range(1,23)
#     ], dtype=np.int32)
dim_input_arr = np.array([len(gene_names),len(ADT_names)])
print(dim_input_arr)


# %%

# Preprocess the data
# X = X.toarray()
print(mods)
print(X)
print(batches)
for i, mm in enumerate(mods):
    for m in mm:
        print(i, mm, m)
        if m=='rna':
                X[batches==i,:] = np.log(X[batches==i,:]/np.sum(X[batches==i,:], axis=1, keepdims=True)*1e4+1.)
        if m=='adt':
                Y[batches==i,:] = np.log(Y[batches==i,:]/np.sum(Y[batches==i,:], axis=1, keepdims=True)*1e4+1.)
# exit()
# Z[Z>0.] = 1.
# Z = Z.toarray()
data = np.c_[X, Y]
print('data shape', data.shape)

# %%
if o.ref_id is None:
    masks = - np.ones((8, np.sum(dim_input_arr)), dtype=np.float32)
else:
    masks = - np.ones((1, np.sum(dim_input_arr)), dtype=np.float32)
for i in a:
    for m in mods[i]:
        print(i, m)
        if m=='rna':
            id = np.where(masks_[i][m]==1)[1]
            masks[i, id] = 0.
        if m=='adt':
            id = np.where(masks_[i][m]==1)[1]
            masks[i, id+len(gene_names)] = 0.
        if m=='ata':
            masks[i, len(gene_names)+len(ADT_names):] = 0.
masks = tf.convert_to_tensor(masks, dtype=tf.float32)
print('mask sum', masks)
print('mask shape', masks.shape)
# %%
from scVAEIT.VAEIT import scVAEIT
path_root = 'result/new_exp/'+task+'/'+o.experiment +'/'

print(path_root)

config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': [len(gene_names),len(ADT_names)],
    'dist_block':['NB','NB'], 
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),
    'dim_block_embed':np.array([16, 8])*2,

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.15,0.85]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
    
}

model = scVAEIT(config, data, masks, batches.reshape(-1, 1))


# %%
# load the model and ensure it is loaded successfully
if o.pretrained is not None:
    print('loading pretrained model')
    checkpoint = tf.train.Checkpoint(net=model.vae)
    epoch = 10
    status = checkpoint.restore(o.pretrained+'checkpoint/ckpt-{}'.format(epoch))
    model.vae(tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
            tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
            tf.zeros((1,np.sum(model.batches.shape[1]))), 
            pre_train=True, L=1, training=True)
    # print(status)

if 'train' in o.actions:
    model.train(
        valid=False, num_epoch=500, batch_size=512, save_every_epoch=50,
        verbose=True, checkpoint_dir=path_root+'checkpoint/', delete_existing=False)

    masks = masks.numpy().copy()
    masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    model.update_z(masks)


# %%
# the latent variables are stored in model.adata
# if 'tea' in task:
#     map_dict = {0:'TEA_w1',1:'TEA_w6',2:'DOGMA_lll_ctrl',3:'DOGMA_dig_stim'}
# else:
#     map_dict = {0:'DOGMA_lll_ctrl',1:'DOGMA_lll_stim',2:'DOGMA_dig_ctrl',3:'DOGMA_dig_stim'}

if not os.path.exists(path_root):
    os.makedirs(path_root)

# if o.ref_id is None:
model.update_z(masks)
pd.DataFrame(model.adata.X).to_csv(path_root+'./P%d_embeddings.csv'%o.ref_id, header=False, index=False)

#     map_dict = {i:'P'+str(i) for i in range(0, 8)}
#     dataset = np.array([map_dict[i] for i in batches])
#     model.adata.obs['Dataset'] = dataset
#     model.adata.obs['Dataset'] = model.adata.obs['Dataset'].astype("category")
#     model.adata.obs['Cell Types'] = cell_types
#     model.adata.write_h5ad(path_root+'/embeddings.h5ad')

# else:
#     # map_dict = {i:'P'+str(i+1) for i in range(0, 8)}
#     # dataset = np.array([[map_dict[i] for i in [0]]])
#     pd.DataFrame(model.adata.X).to_csv(path_root+'./P'+str(o.ref_id+1)+'_embeddings.csv', header=False, index=False)
#     model.adata.obs['Dataset'] = 'P'+str(o.ref_id+1)
#     model.adata.obs['Dataset'] = model.adata.obs['Dataset'].astype("category")
#     model.adata.obs['Cell Types'] = cell_types

#     pd.DataFrame(model.adata.X).to_csv(path_root+'./P'+str(o.ref_id+1)+'_embeddings.csv', header=False, index=False)
#     model.adata.write_h5ad(path_root+'./P'+str(o.ref_id+1)+'_embeddings.h5ad')

#     sc.tl.umap(model.adata)
#     sc.pl.umap(model.adata, color = ['Cell Types'], save='_'+task+'_'+o.experiment+'_P'+str(o.ref_id+1)+'_scVAEIT_merged_label.png', show=False)
#     sc.pl.umap(model.adata, color = ['Dataset'], save='_'+task+'_'+o.experiment+'_P'+str(o.ref_id+1)+'_scVAEIT_merged_batch.png', show=False)

if 'recon' in o.actions:
    print('generating recon')
    denoised_data = model.get_denoised_data()
    if o.ref_id is None:
        pd.DataFrame(denoised_data).to_csv(path_root+'/recon.csv')
    else:
        pd.DataFrame(denoised_data).to_csv(path_root+'P%d_recon.csv'%(o.ref_id+1))


# %%

# model.adata.write_h5ad(path_root+'/joint.h5ad')



