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
# from memory_profiler import profile
import scanpy as sc
import tensorflow as tf
import tensorflow_probability as tfp
# import psutil
# def process_memory():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info.rss
 
# decorator function
# def profile(func):
#     def wrapper(*args, **kwargs):
 
#         mem_before = process_memory()
#         result = func(*args, **kwargs)
#         mem_after = process_memory()
#         print("{}:consumed memory: {:,}".format(
#             func.__name__,
#             mem_before, mem_after, mem_after - mem_before))
 
#         return result
#     return wrapper

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

# %%
import argparse
parser = argparse.ArgumentParser()
## Task
parser.add_argument('--task', type=str, default='teadog',
    help="Choose a task")
parser.add_argument('--experiment', type=str, default='offline',
    help="")
parser.add_argument('--ref_id', type=int, default=None,
    help="reference task id, used in the transfer comparison. If ref_id is None, it is the offline version of scvaeit")
parser.add_argument('--epoch', type=int, default=500,
    help="")
parser.add_argument('--pretrain_ep', type=int, default=10,
    help="")
parser.add_argument('--pretrained', type=str, default=None,
    help="pretrained model path")
parser.add_argument('--actions', type=str, default=['train'], nargs='+',
    help="")
o = parser.parse_args()
task = o.task
data_root = '/dev/shm/processed/'+task

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
    mask = {}
    for f in os.listdir(p+'/mat'):
        print(i,f)
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
batches = np.concatenate(batches)
# cell_types = np.concatenate(cell_types)[:, 1]

# %%
gene_names = pd.read_csv(data_root+'/feat/feat_names_rna.csv',header=0, index_col=0)
peak_names = pd.read_csv(data_root+'/feat/feat_names_atac.csv',header=0, index_col=0)
ADT_names = pd.read_csv(data_root+'/feat/feat_names_adt.csv',header=0, index_col=0)

# %%
chunk_atac = np.array([
    np.sum(np.char.startswith(list(peak_names['x']), 'chr%d-'%i)) for i in range(1,23)
    ], dtype=np.int32)

# %%

# %%
X = np.zeros((n_cells, len(gene_names)))
Y = np.zeros((n_cells, len(ADT_names)))
Z = np.zeros((n_cells, len(peak_names)))
print(mods)
for i, mm in enumerate(mods):
    for m in mm:
        if m=='rna':
            X[n_cells_list[i]:n_cells_list[i+1], :] = data[i][m]
        if m=='adt':
            Y[n_cells_list[i]:n_cells_list[i+1], :] = data[i][m]
        if m=='ata':
            Z[n_cells_list[i]:n_cells_list[i+1], :] = data[i][m]

# del data
# %%
# Count peaks in each chromosome (assuming they are ordered)
chunk_atac = np.array([
    np.sum(np.char.startswith(list(peak_names['x']), 'chr%d-'%i)) for i in range(1,23)
    ], dtype=np.int32)
dim_input_arr = np.array([len(gene_names),len(ADT_names),len(peak_names)])
print(dim_input_arr)


# %%

# Preprocess the data
# X = X.toarray()
for i, mm in enumerate(mods):
    for m in mm:
        if m=='rna':
            X[batches==i,:] = np.log(X[batches==i,:]/np.sum(X[batches==i,:], axis=1, keepdims=True)*1e4+1.)
        if m=='adt':
            Y[batches==i,:] = np.log(Y[batches==i,:]/np.sum(Y[batches==i,:], axis=1, keepdims=True)*1e4+1.)
Z[Z>0.] = 1.
# Z = Z.toarray()
data = np.c_[X, Y, Z]
print(data)

# %%
if o.ref_id is None:
    masks = - np.ones((8, np.sum(dim_input_arr)), dtype=np.float32)
else:
    masks = - np.ones((1, np.sum(dim_input_arr)), dtype=np.float32)
for i in a:
    for m in mods[i]:
        print('mask', i, m)
        if m=='rna':
            id = np.where(masks_[i][m]==1)[1]
            masks[i, id] = 0.
        if m=='adt':
            id = np.where(masks_[i][m]==1)[1]
            masks[i, id+len(gene_names)] = 0.
        if m=='ata':
            masks[i, len(gene_names)+len(ADT_names):] = 0.
masks = tf.convert_to_tensor(masks, dtype=tf.float32)
print('mask sum', sum(masks))
print('mask shape', masks.shape)

# %%
from scVAEIT.VAEIT import scVAEIT
path_root = 'result/new_exp/'+task+'/'+o.experiment +'/'

config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256], 
    'dim_latent':32,
    'dim_block': np.append([len(gene_names),len(ADT_names)], chunk_atac), 
    'dist_block':['NB','NB'] + ['Bernoulli' for _ in chunk_atac], 
    'dim_block_enc':np.array([256, 128] + [16 for _ in chunk_atac]),
    'dim_block_dec':np.array([256, 128] + [16 for _ in chunk_atac]),
    'block_names':np.array(['rna', 'adt'] + ['atac' for _ in range(len(chunk_atac))]),
    'uni_block_names':np.array(['rna','adt','atac']),
    'dim_block_embed':np.array([16, 8] + [1 for _ in range(len(chunk_atac))])*2,

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.14,0.85,0.01]),
    'beta_reverse':0.,

    "p_feat" : 0.2,
    "p_modal" : np.ones(3)/3,
    
}

model = scVAEIT(config, data, masks, batches.reshape(-1, 1))


# checkpoint = tf.train.Checkpoint(net=model.vae)
# epoch = 5
# status = checkpoint.restore(path_root+'checkpoint/ckpt-{}'.format(epoch))
# model.vae(tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
#           tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
#           tf.zeros((1,np.sum(model.batches.shape[1]))), 
#           pre_train=True, L=1, training=False)
# print(status)
if o.pretrained is not None:
    print('loading pretrained model')
    checkpoint = tf.train.Checkpoint(net=model.vae)
    epoch = o.pretrain_ep
    status = checkpoint.restore(o.pretrained+'checkpoint/ckpt-{}'.format(epoch))
    model.vae(tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
            tf.zeros((1,np.sum(model.vae.config.dim_input_arr))),
            tf.zeros((1,np.sum(model.batches.shape[1]))), 
            pre_train=True, L=1, training=True)
# %%

# @profile
def train():
    model.train(
        valid=False, num_epoch=o.epoch, batch_size=512, save_every_epoch=50,
        verbose=True, checkpoint_dir=path_root+'checkpoint/', delete_existing=False, random_state=42)
    

if 'train' in o.actions:
    train()

# %%
masks = masks.numpy().copy()
masks = tf.convert_to_tensor(masks, dtype=tf.float32)
model.update_z(masks)

pd.DataFrame(model.adata.X).to_csv(path_root+'./embeddings.csv')
