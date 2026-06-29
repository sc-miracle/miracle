# %%
import argparse
from modules import models, utils
import os
from os import path
from os.path import join as pj
import numpy as np
import torch as th
import math
from torch.utils.data import Dataset
import re
import pandas as pd
import copy
from torch import nn, autograd
import re
import itertools
from tqdm import tqdm
import random
from sklearn.neighbors import BallTree
import time

# %% [markdown]
# # Config

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='atlas',
    help="Choose a task")
parser.add_argument('--experiment', type=str, default='e1',
    help="Choose an experiment")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
parser.add_argument('--actions', type=str, nargs='+', default=['train', 'subsample', 'predict_subsample', 'predict_all'],
    help="Choose actions to run")
parser.add_argument('--method', type=str, default='midas',
    help="Choose an method to benchmark")
parser.add_argument('--init_model', type=str, default='',
    help="Load a trained model")
parser.add_argument('--epoch_num', type=int, default=2000,
    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument('--grad_clip', type=float, default=-1,
    help='Gradient clipping value')
parser.add_argument('--s_drop_rate', type=float, default=0.1,
    help="Probility of dropping out subject ID during training")
parser.add_argument('--drop_s', type=int, default=0,
    help="Force to drop s")
parser.add_argument('--seed', type=int, default=0,
    help="Set the random seed to reproduce the results")
parser.add_argument('--use_shm', type=int, default=0,
    help="Use shared memory to accelerate training")
## Debugging
parser.add_argument('--print_iters', type=int, default=-1,
    help="Iterations to print training messages")
parser.add_argument('--log_epochs', type=int, default=500,
    help='Epochs to log the training states')
parser.add_argument('--save_epochs', type=int, default=10,
    help='Epochs to save the latest training states (overwrite previous ones)')
parser.add_argument('--time', type=int, default=0, choices=[0, 1],
    help='Time the forward and backward passes')
parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
    help='Print intermediate variables')

## Continual Learning
parser.add_argument('--data_current', type=str, nargs='+', default=[],
    help="Task name of current data")
parser.add_argument('--data_replay', type=str, nargs='+', default=[],
    help="Task name of replay data")
parser.add_argument('--model_path', type=str, default='',
    help="Load model weights from this path")
parser.add_argument('--last_subsample_dir', type=str, default='',
    help="Read subsample files from this dir")
parser.add_argument('--max_size', type=int, default=100000,
    help="The max storage size of samples.")
parser.add_argument('--use_predefined_mods', type=int, default=1,
    help="Use predefine mods in continual task")
parser.add_argument('--REPLAY', type=int, default=1,
    help="")
parser.add_argument('--subsample_method', type=str, default='bts',
    help="")
parser.add_argument('--adv', type=float, default=30,
    help="")
o = parser.parse_args()


# %%
def get_dataloaders_cl(datasets, shuffle=True):
    if o.REPLAY:
        dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
        # define a sampler for continual learning: output current data and replay data iteratly
        # if there is one batch of current data, output: c r1 c r2 c r3
        # if there are more than one batches of current data, output: c1 r1 c2 r2 c3 r3
        sampler = MultiDatasetSampler2(dataset_cat, o.current_num, batch_size=o.N, shuffle=shuffle)
        data_loader = th.utils.data.DataLoader(dataset_cat, batch_size=o.N, sampler=sampler, 
            num_workers=64, pin_memory=True)
    else:
        print('using only current data')
        data_loader = th.utils.data.DataLoader(datasets[-1], batch_size=o.N, 
        num_workers=64, pin_memory=True)
        print(o.replay_num, o.current_num)
    return data_loader

# a common dataloaders that output datasets one by one
def get_dataloaders(datasets, shuffle=True):
    # dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
    data_loader = {}
    for i, dataset in enumerate(datasets):
        data_loader[i] = th.utils.data.DataLoader(dataset, batch_size=o.N, 
        num_workers=64, pin_memory=True)
    return data_loader

# %%
def run_epoch(data_loader, split, epoch_id=0):
    start_time = time.time()

    if split == "train":
        net.train()
        discriminator.train()
    elif split == "test":
        net.eval()
        discriminator.eval()
    else:
        assert False, "Invalid split: %s" % split

    loss_total = 0
    for i, data in enumerate(data_loader):
        # print(i,data['s']['joint'][0])
        if not o.REPLAY:
            rnt_ = 1
        # for replay and current data, different rnt should be used to balance their training frequency
        elif  o.replay_num == 0 or i%2 < 1:
            # for current data
            rnt_ = o.n_cells_orig[data['s']['joint'][0].item()]/sum(o.n_cells_orig)
        else:
            # for replay data
            rnt_ = o.n_cells_orig[data['s']['joint'][0].item()]/sum(o.n_cells_orig) * o.replay_num
        loss = run_iter(split, epoch_id, data, rnt_)
        loss_total += loss
        if o.print_iters > 0 and (i+1) % o.print_iters == 0:
            print('%s\tepoch: %d/%d\tBatch: %d/%d\t%s_loss: %.3f'.expandtabs(3) % 
                  (o.task, epoch_id+1, o.epoch_num, i+1, len(data_loader), split, loss))

    loss_avg = loss_total / len(data_loader)
    epoch_time = (time.time() - start_time) / 3600 / 24
    elapsed_time = epoch_time * (epoch_id+1)
    total_time = epoch_time * o.epoch_num
    print('%s\t%s\tepoch: %d/%d\t%s_loss: %.2f\ttime: %.1f/%.1f\n'.expandtabs(3) % 
          (o.task, o.experiment, epoch_id+1, o.epoch_num, split, loss_avg, elapsed_time, total_time))
    benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    return loss_avg

def run_iter(split, epoch_id, inputs, rnt=1):
    inputs = utils.convert_tensors_to_cuda(inputs)
    # print('inputs', inputs['s']['joint'][0])
    if split == "train":
        with autograd.set_detect_anomaly(o.debug == 1):
            loss_net, c_all = forward_net(inputs)
            discriminator.epoch = epoch_id - o.ref_epoch_num
            K = 3
            for _ in range(K):
                loss_disc = forward_disc(utils.detach_tensors(c_all), inputs["s"])
                loss_disc = loss_disc * rnt
                update_disc(loss_disc)
            # c = models.CheckBP('c')(c)
            loss_adv = forward_disc(c_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv
            loss = rnt * loss
            update_net(loss)
        
            
    else:
        with th.no_grad():
            loss_net, c_all = forward_net(inputs)
            loss_adv = forward_disc(c_all, inputs["s"])
            loss_adv = -loss_adv
            loss = loss_net + loss_adv

    return loss.item()

def forward_net(inputs):
    return net(inputs)


def forward_disc(c, s):
    return discriminator(c, s)


def update_net(loss):
    update(loss, net, optimizer_net)


def update_disc(loss):
    update(loss, discriminator, optimizer_disc)
    

def update(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()
    if o.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    optimizer.step()


def check_to_save(epoch_id):
    if (epoch_id+1) % o.log_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_%08d" % epoch_id)
    if (epoch_id+1) % o.save_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_latest")


def save_training_states(epoch_id, filename):
    benchmark['epoch_id_start'] = epoch_id + 1
    utils.save_toml({"o": vars(o), "benchmark": benchmark}, pj(o.train_dir, filename+".toml"))
    th.save({"net_states": net.state_dict(),
             "disc_states": discriminator.state_dict(),
             "optim_net_states": optimizer_net.state_dict(),
             "optim_disc_states": optimizer_disc.state_dict()
            }, pj(o.train_dir, filename+".pt"))


def predict(datasets, joint_latent=True, mod_latent=False, impute=False, batch_correct=False, translate=False, 
            input=False):
    if translate:
        mod_latent = True
    print("Predicting ...")
    dirs = utils.get_pred_dirs(o, joint_latent, mod_latent, impute, batch_correct, translate, input)
    parent_dirs = list(set(map(path.dirname, utils.extract_values(dirs))))
    utils.mkdirs(parent_dirs, remove_old=True)
    utils.mkdirs(dirs, remove_old=True)
    data_loaders = get_dataloaders(datasets, shuffle=False)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
            for i, data in enumerate(tqdm(data_loader)):
                data = utils.convert_tensors_to_cuda(data)
                
                # conditioned on all observed modalities
                if joint_latent:
                    x_r_pre, _, _, _, z, _, _, *_ = net.sct(data)  # N * K
                    utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                if impute:
                    x_r = models.gen_real_data(x_r_pre, sampling=False)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_impt"][m], fname_fmt) % i)
                if input:  # save the input
                    for m in o.combs[subset_id]:
                        utils.save_tensor_to_csv(data["x"][m].int(), pj(dirs[subset_id]["x"][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if mod_latent:
                    for m in data["x"].keys():
                        input_data = {
                            "x": {m: data["x"][m]},
                            "s": data["s"], 
                            "e": {}
                        }
                        if m in data["e"].keys():
                            input_data["e"][m] = data["e"][m]
                        x_r_pre, _, _, _, z, c, b, *_ = net.sct(input_data)  # N * K
                        utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"][m], fname_fmt) % i)
                        if translate: # single to double
                            x_r = models.gen_real_data(x_r_pre, sampling=False)
                            for m_ in set(o.mods) - {m}:
                                utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m+"_to_"+m_], fname_fmt) % i)
                
                if translate: # double to single
                    for mods in itertools.combinations(data["x"].keys(), 2):
                        m1, m2 = utils.ref_sort(mods, ref=o.mods)
                        input_data = {
                            "x": {m1: data["x"][m1], m2: data["x"][m2]},
                            "s": data["s"], 
                            "e": {}
                        }
                        for m in mods:
                            if m in data["e"].keys():
                                input_data["e"][m] = data["e"][m]
                        x_r_pre, *_ = net.sct(input_data)  # N * K
                        x_r = models.gen_real_data(x_r_pre, sampling=False)
                        m_ = list(set(o.mods) - set(mods))[0]
                        utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m1+"_"+m2+"_to_"+m_], fname_fmt) % i)

        if batch_correct:
            print("Calculating b_centroid ...")
            # z, c, b, subset_ids, batch_ids = utils.load_predicted(o)
            # b = th.from_numpy(b["joint"])
            # subset_ids = th.from_numpy(subset_ids["joint"])
            
            pred = utils.load_predicted(o)
            b = th.from_numpy(pred["z"]["joint"][:, o.dim_c:])
            s = th.from_numpy(pred["s"]["joint"])

            b_mean = b.mean(dim=0, keepdim=True)
            b_subset_mean_list = []
            for subset_id in s.unique():
                b_subset = b[s == subset_id, :]
                b_subset_mean_list.append(b_subset.mean(dim=0))
            b_subset_mean_stack = th.stack(b_subset_mean_list, dim=0)
            dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
            net.sct.b_centroid = b_subset_mean_list[dist.argmin()]
            net.sct.batch_correction = True
            
            print("Batch correction ...")
            for subset_id, data_loader in data_loaders.items():
                print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
                fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = utils.convert_tensors_to_cuda(data)
                    x_r_pre, *_ = net.sct(data)
                    x_r = models.gen_real_data(x_r_pre, sampling=True)
                    for m in o.mods:
                        utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_bc"][m], fname_fmt) % i)

def subsample(pred_dir='', size=100000, remove_old=True):
    o.subsample_dir = pj(o.result_dir, 'subsample')
    utils.mkdir(o.subsample_dir, remove_old=remove_old)
    cell_num = []
    for i in data_config['cell_names_path_ref']:
        cell_num.append(len(utils.load_csv(i))-1)
    
    for i in range(o.current_num + o.replay_num):
        s = int(cell_num[i] / sum(cell_num) * size)
        nn = len(utils.load_csv(data_config['cell_names_path'][i]))-1
        # print(s)
        if o.subsample_method == 'bts':
            # for p in os.listdir(pj(pred_dir)):
            # i = int(re.sub('subset_','', p))
            
            preds = []
            for j in os.listdir(pj(pred_dir, 'subset_%d'%i, 'z', 'joint')):
                preds += utils.load_csv(pj(pred_dir, 'subset_%d'%i, 'z', 'joint', j))
            preds = np.array(preds)
            n = BallTreeSubsample(preds[:, :32], s)
            n.sort()
        else:
            n = list(range(nn))
            random.shuffle(n)
            n = n[:s]
            n.sort()
        cell_names = np.array(utils.load_csv(data_config['cell_names_path'][i]))[1:]
        save_path = pj(o.subsample_dir, 'subset_%d.csv'%i)
        pd.DataFrame(cell_names[n]).to_csv(save_path, index=False)
        print('%s subsample subset %d from %d to %d and save cell names in %s' % (o.subsample_method, i, nn, len(n), save_path))

def BallTreeSubsample(X, target_size, ls=2):

    if target_size >= len(X):
        return list(range(len(X)))
    
    # construct a tree: nodes and corresponding order list of samples
    tree = BallTree(X, leaf_size = ls)

    layer = int(np.log2(len(X)//ls))
    t = [1]
    for i in range(layer+1):
        t.append(t[i]*2)
    
    t = [i-1 for i in t]
    t.sort(reverse=True)
    nodes = tree.get_arrays()[2]
    order = tree.get_arrays()[1]
    target = []

    # subsample in a bottom-up order
    # from the bottom of the tree to the top
    for l in range(layer):
        if len(target) < target_size:
            s = (target_size - len(target)) // (t[l:l+2][0]- t[l:l+2][1])
        else:
            return target
        for node in nodes[t[l:l+2][1]:t[l:l+2][0]]:
            start_id = node[0]
            end_id = node[1]
            available_order = list(set(order[start_id:end_id])-set(target))
            random.shuffle(available_order)
            target.extend(available_order[0:s])
    return target


# %%
def get_info_from_dir(path, config, subset, mods_control=None, comb_ratios=1):
    dirs = os.listdir(path)
    n = 0
    mods_type = []
    for f in dirs:
        
        if f=='feat':
            assert 'feat_dims.csv' in os.listdir(pj(path, f))
        elif 'subset' in f :
            config[n+subset] = {'feat':[], 'combs':[], 's_joint':[]}
            if mods_control is None:
                # use default mods
                config[n+subset]['combs'] = []
                for m in os.listdir(pj(path, f, 'mat')):
                    mod = m[:-len('.csv')]
                    config[n+subset]['combs'].append(mod)
            else:
                # use mods input
                config[n+subset]['combs'] = mods_control
            for m in config[n+subset]['combs']:
                if m not in mods_type:
                    mods_type.append(m)
            config[n+subset]['feat'] = {}
            
            # feat['feat_dims'] = pd.read_csv(pj(path,  'feat', 'feat_dims.csv'))[mods_type]
            for s in mods_type:
                print(n+subset, pj(path, 'feat', 'feat_names_%s.csv' %s))
                print(list(np.array(utils.load_csv(pj(path, 'feat', 'feat_names_%s.csv' %s)))[1:10, 1]))
                assert 'feat_names_%s.csv' %s in os.listdir(pj(path,'feat'))
                config[n+subset]['feat'][s] = list(np.array(utils.load_csv(pj(path, 'feat', 'feat_names_%s.csv' %s)))[1:, 1])
                print(config[n+subset]['feat'][s][1:10])

            config[n+subset]['s_joint'] = [n+subset]
            config[n+subset]['comb_ratios'] = [1]
            n += 1
        print('%d batches detected in dir %s' % (n, path))
    return n, config

# %%
def combine_set(u,v):
    temp = copy.deepcopy(u)
    dif = difference(v, temp)
    temp.extend(dif)
    return temp

def difference(v, u):
    # print(len(v), len(u))
    temp = []
    for i in v:
        if i not in u:
            temp.append(i)
    return temp
    
# def get_common(a,b):
#     return combine_set(a, b)
def gen_map(u, v):
    index1 = []
    index2 = []
    for i, j in enumerate(u):
        if j in v:
            index1.append(i)
            index2.append(v.index(j))
    temp = [index1, index2]
    return temp


# %%
def initialize():
    init_seed()
    init_dirs()
    load_data_config()
    load_model_config()
    get_gpu_config()
    init_model()


def init_seed():
    if o.seed >= 0:
        np.random.seed(o.seed)
        th.manual_seed(o.seed)
        th.cuda.manual_seed_all(o.seed)


def init_dirs():
    data_folder = re.sub("_generalize", "_transfer", o.task)
    if o.use_shm == 1:
        o.data_dir = pj("/dev/shm", "processed", data_folder)
    else:
        o.data_dir = pj("data", "processed", data_folder)
    o.result_dir = pj("result", o.task, o.experiment, o.model)
    o.pred_dir = pj(o.result_dir, "predict", o.init_model)
    o.train_dir = pj(o.result_dir, "train")
    o.debug_dir = pj(o.result_dir, "debug")
    utils.mkdirs([o.train_dir, o.debug_dir])
    # print("Task: %s\nExperiment: %s\nModel: %s\n" % (o.task, o.experiment, o.model))


def load_data_config():
    
    # if o.reference == '':
    o.dims_x, o.dims_chr, o.mods = get_dims_x(ref=0)
    o.ref_mods = o.mods
    # else:
    #     _, _, o.mods = get_dims_x(ref=0)
    #     o.dims_x, o.dims_chr, o.ref_mods = get_dims_x(ref=1)
    o.mod_num = len(o.mods)
    
    global data_config

    for k, v in data_config.items():
        vars(o)[k] = v

    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)

    o.dim_s = o.dims_s["joint"]
    o.dim_b = 2



def load_model_config():
    model_config = utils.load_toml("configs/model.toml")["default"]
    if o.model != "default":
        model_config.update(utils.load_toml("configs/model.toml")[o.model])
    for k, v in model_config.items():
        vars(o)[k] = v
    o.dim_z = o.dim_c + o.dim_b
    o.dims_dec_x = o.dims_enc_x[::-1]
    o.dims_dec_s = o.dims_enc_s[::-1]
    if "dims_enc_chr" in vars(o).keys():
        o.dims_dec_chr = o.dims_enc_chr[::-1]
    o.dims_h = {}
    for m, dim in o.dims_x.items():
        o.dims_h[m] = dim if m != "atac" else o.dims_enc_chr[-1] * 22


def get_gpu_config():
    o.G = 1  # th.cuda.device_count()  # get GPU number
    assert o.N % o.G == 0, "Please ensure the mini-batch size can be divided " \
        "by the GPU number"
    o.n = o.N // o.G
    print("Total mini-batch size: %d, GPU number: %d, GPU mini-batch size: %d" % (o.N, o.G, o.n))


def init_model():
    """
    Initialize the model, optimizer, and benchmark
    """
    global net, discriminator, optimizer_net, optimizer_disc
    
    # Initialize models
    net = models.Net(o).cuda()
    discriminator = models.Discriminator(o).cuda()
    net_param_num = sum([param.data.numel() for param in net.parameters()])
    disc_param_num = sum([param.data.numel() for param in discriminator.parameters()])
    print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))
    
    # Load benchmark
    if o.init_model != '':
        # if o.init_from_ref == 0:
        fpath = pj(o.train_dir, o.init_model)
        savepoint_toml = utils.load_toml(fpath+".toml")
        benchmark.update(savepoint_toml['benchmark'])
        o.ref_epoch_num = savepoint_toml["o"]["ref_epoch_num"]
        # else:
        #     fpath = pj("result", o.task, o.experiment, o.model, "train", o.init_model)
        #     benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
        #     o.ref_epoch_num = benchmark["epoch_id_start"]
    else:
        o.ref_epoch_num = 0
    # Initialize optimizers
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    optimizer_disc = th.optim.AdamW(discriminator.parameters(), lr=o.lr)
    
    # Load models and optimizers
    if o.init_model != '':
        savepoint = th.load(fpath+".pt")
        # if o.init_from_ref == 0:
        net.load_state_dict(savepoint['net_states'])
        discriminator.load_state_dict(savepoint['disc_states'])
        optimizer_net.load_state_dict(savepoint['optim_net_states'])
        optimizer_disc.load_state_dict(savepoint['optim_disc_states'])
        # else:
        #     exclude_modules = ["s_enc", "s_dec"]
        #     pretrained_dict = {}
        #     for k, v in savepoint['net_states'].items():
        #         exclude = False
        #         for exclude_module in exclude_modules:
        #             if exclude_module in k:
        #                 exclude = True
        #                 break
        #         if not exclude:
        #             pretrained_dict[k] = v
        #     net_dict = net.state_dict()
        #     net_dict.update(pretrained_dict)
        #     net.load_state_dict(net_dict)
        print('Model is initialized from ' + fpath + ".pt")


def print_model():
    global net, discriminator
    with open(pj(o.result_dir, "model_architecture.txt"), 'w') as f:
        print(net, file=f)
        print(discriminator, file=f)


def get_dims_x(ref):
    # if o.cl:
    feat_dims = [['place_holder']]
    for k in data_config['feat_dims'].keys():
        # print(k, data_config['feat_dims'][k])
        feat_dims.append([k] + data_config['feat_dims'][k])
    # elif ref == 0:
    #     feat_dims = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    #     feat_dims = utils.transpose_list(feat_dims)
    # else:
    #     feat_dims = utils.load_csv(pj("data", "processed", o.reference, "feat", "feat_dims.csv"))
    #     feat_dims = utils.transpose_list(feat_dims)
    # print(feat_dims)
    dims_x = {}
    dims_chr = []
    for i in range(1, len(feat_dims)):
        m = feat_dims[i][0]
        if m == "atac":
            dims_chr = list(map(int, feat_dims[i][1:]))
            dims_x[m] = sum(dims_chr)
        else:   
            dims_x[m] = int(feat_dims[i][1])
    print("Input feature numbers: ", dims_x)

    mods = list(dims_x.keys())

    return dims_x, dims_chr, mods

# %%

# This is an important step for continual learning
# 1. get all features of current data and replay data
# 2. combine these featrues by training order
# o.trainsfrom is used in constructing dataloader, which help reconstructing training data by new feature order.

def combine_data():
    global data_config
    o.transform = {}
    n = 0
    o.current_num = 0
    for i, task in enumerate(o.data_replay + o.data_current):
        # print('combining task : %s' % (task))
        if o.use_shm == 1:
            data_dir = pj("/dev/shm", "processed", task)
        else:
            data_dir = pj("data", "processed", task)

        if i==0:
            data_config = utils.load_toml("configs/data.toml")[task]
       
            for k, v in data_config.items():
                # print(v)
                data_config[k] = v if type(v)==list else v
                
            # print(data_config)
            data_config['data_dir'] = []
            data_config['subset'] = []
            data_config['cell_names_path'] = []
            data_config['cell_names_path_ref'] = []
            # print(data_config)

        else:
            data_config_ref = utils.load_toml("configs/data.toml")[task]
            for k, v in data_config_ref.items():
                data_config_ref[k] = v if type(v)==list else v
            data_config["raw_data_dirs"] += data_config_ref["raw_data_dirs"]
            data_config["raw_data_frags"] += data_config_ref["raw_data_frags"]
            data_config["combs"] += data_config_ref["combs"]
            data_config["comb_ratios"] += data_config_ref["comb_ratios"]
            data_config["s_joint"] += [[v[0]+len(data_config["s_joint"])] for v in data_config_ref["s_joint"]]
            # print(data_config)
 
        # j = 0
        all_data_dir = {}
        for ii in sorted(os.listdir(pj(data_dir))):
            if 'subset' in ii: 
                all_data_dir[ii.split('_')[1]] = ii
        
        for j in range(len(all_data_dir)):
            p = all_data_dir[str(j)]
            # if 'subset' in :
            # print(p)
            data_config['data_dir'].append(data_dir)
            data_config['subset'].append(j)
            if o.last_subsample_dir != '' :
                if os.path.exists(pj(o.last_subsample_dir, 'subset_%d.csv'%n)):
                    data_config['cell_names_path'].append(pj(o.last_subsample_dir, 'subset_%d.csv'%n))
                else:
                    data_config['cell_names_path'].append(pj(data_dir, p, 'cell_names.csv'))
            else:
                data_config['cell_names_path'].append(pj(data_dir, p, 'cell_names.csv'))
            data_config['cell_names_path_ref'].append(pj(data_dir, p, 'cell_names.csv'))
            o.transform[str(n)] = {}
            n += 1
            # j += 1
            if i >= len(o.data_replay):
                o.current_num += 1
    n_cells_orig = []
    for i in data_config['cell_names_path_ref']:
        n_cells_orig.append(len(utils.load_csv(i))-1)
    o.n_cells_orig = n_cells_orig
    # if o.control_mods != []:
    #     data_config["combs"] = o.control_mods
    # else:

    
    if o.use_predefined_mods:
        data_config['combs'] = utils.load_toml("configs/data.toml")[o.task]['combs'][:n]
        data_config["combs"] = [k[0] for k in data_config["combs"]]
        # print(data_config['combs'])
    else:
        data_config["combs"] = [k[0] for k in data_config["combs"]]
        # print(data_config['combs'])
    # print('mods:')
    print('MODS:', *data_config['combs'])
    data_config['feat'] = {}
    data_config['feat_dims'] = {}


    for i in range(n):
        for m in data_config["combs"][i]:
            feat = list(np.array(utils.load_csv(pj(data_config['data_dir'][i], 'feat', 'feat_names_%s.csv'%m)))[1:, 1])
            # print(feat[1:10], len(feat))
            if m not in data_config['feat'].keys() and m!='atac':
                data_config['feat'][m] = feat
            elif m != 'atac':
                data_config['feat'][m] = combine_set(data_config['feat'][m], feat)
            if m!= 'atac':
                data_config['feat_dims'][m] = [len(data_config['feat'][m])]
            else:
                data_config['feat_dims'][m] = list(pd.read_csv(pj(data_config['data_dir'][i], 'feat', 'feat_dims.csv'), index_col=0)['atac'])
            if m!='atac':
                o.transform[str(i)][m] = gen_map(data_config['feat'][m], feat)
                # print(i, m, len(feat), np.array(o.transform[str(i)][m]).shape, np.array(o.transform[str(i)][m])[1].max())
    data_config['combs'] = [data_config['combs']]
    o.replay_num = n - o.current_num




# %%
#
def update_state_by_position(model, past_model, model_name):
    temp_model = model.state_dict()
    loaded = 0
    partial = 0
    skipped = 0
    for k, v in temp_model.items():
        if k not in past_model.keys():
            skipped += 1
            continue
        past_tensor = past_model[k].to(device=v.device, dtype=v.dtype)
        if v.shape == past_tensor.shape:
            temp_model[k] = past_tensor
            loaded += 1
            continue
        if len(v.shape) == len(past_tensor.shape):
            slices = tuple(slice(0, min(v.shape[i], past_tensor.shape[i])) for i in range(len(v.shape)))
            temp_model[k][slices] = past_tensor[slices]
            partial += 1
        else:
            skipped += 1
    model.load_state_dict(temp_model)
    print('update %s: loaded=%d, partial=%d, skipped=%d' % (model_name, loaded, partial, skipped))


def update_model(model_path, model, disc_model=None):
    # if o.init_model != '':
    print('update model with weights from %s' % model_path)
    # print(o.dims_h)
    savepoint = th.load(model_path)
    # print(model_path[:-3]+'.toml')
    # print(utils.load_toml(model_path[:-3]+'.toml')['o']['dims_h'])
    dims_h_past = utils.load_toml(model_path[:-3]+'.toml')['o']['dims_h']
    # print(dims_h_past)
    past_model = savepoint['net_states']
    temp_model = model.state_dict()
    for k, v in temp_model.items():
        if k not in past_model.keys():
            continue
        past_tensor = past_model[k].to(device=v.device, dtype=v.dtype)
        if k == 'sct.x_dec.net.4.weight' or k == 'sct.x_dec.net.4.bias':
            param_dict_last = dict(zip(dims_h_past.keys(), past_tensor.split(list(dims_h_past.values()), dim=0)))
            param_dict_len = [i.shape[0] for i in temp_model[k].split(list(o.dims_h.values()), dim=0)]
            shape_list = dict(zip(o.dims_h.keys(), param_dict_len))
            start_id = 0
            # enhanced to avoid problems when doing transfer learning
            for m in o.dims_h.keys():
                if m in dims_h_past.keys():
                    temp_model[k][start_id:start_id+param_dict_last[m].shape[0]] = param_dict_last[m]
                start_id += shape_list[m]
        elif v.shape == past_tensor.shape:
            temp_model[k] = past_tensor
        elif len(v.shape)==2 and v.shape[0] == past_tensor.shape[0]:
            # print(k, past_model[k].shape, temp_model[k].shape)
            temp_model[k][:, :past_tensor.shape[1]] = past_tensor
        elif len(v.shape)==2 and v.shape[1] == past_tensor.shape[1]:
            # print(k, past_model[k].shape, temp_model[k].shape)
            temp_model[k][:past_tensor.shape[0], :] = past_tensor
        else:
            # print(k, past_model[k].shape, temp_model[k].shape)
            temp_model[k][:past_tensor.shape[0]] = past_tensor
    model.load_state_dict(temp_model)

    if disc_model is not None:
        if 'disc_states' in savepoint.keys():
            update_state_by_position(disc_model, savepoint['disc_states'], 'discriminator')
        else:
            print('skip discriminator update: disc_states not found in %s' % model_path)

# %%
class MultimodalDataset(Dataset):

    def __init__(self, config, subset=0, transform=None, split="train", comb=None, train_ratio=None, subsample=None):
        super(MultimodalDataset, self).__init__()
        
        # config = utils.gen_data_config(task)
        # for kw, arg in config.items():
        #     setattr(self, kw, arg)    
        self.combs = config['combs']
        self.s_joint = config['s_joint']
        _, self.combs, self.s, _ = utils.gen_all_batch_ids(self.s_joint, self.combs)
        assert subset < len(self.combs) == len(self.s), "Inconsistent subset specifications!"
        self.subset = subset
        self.comb = self.combs[subset] if comb is None else comb
        if train_ratio is not None: self.train_ratio = train_ratio
        self.s_subset = self.s[subset]
        self.train_ratio = config['train_ratio']
        data_dir = config['data_dir'][subset]
        # self.s_drop_rate = s_drop_rate if split == "train" else 0
        base_dir = pj(data_dir, "subset_"+str(config['subset'][subset]))
        # print(base_dir)
        self.in_dirs = {}
        self.masks = {}
        for m in self.comb:
            self.in_dirs[m] = pj(base_dir, "vec", m)
            if m in ["rna", "adt"]:
                mask = utils.load_csv(pj(base_dir, "mask", m+".csv"))[1][1:]
                self.masks[m] = np.array(mask, dtype=np.float32)

        filenames_list = []
        for in_dir in self.in_dirs.values():
            filenames_list.append(utils.get_filenames(in_dir, "csv"))
        cell_nums = [len(filenames) for filenames in filenames_list]
        assert cell_nums[0] > 0 and len(set(cell_nums)) == 1, \
            "Inconsistent cell numbers!"
        filenames = filenames_list[0]
        if subsample is not None:
            # print(len(filenames), max(subsample))
            filenames = list(np.array(filenames)[subsample])

        train_num = int(round(len(filenames) * self.train_ratio))
        if split == "train":
            self.filenames = filenames[:train_num]
        else:
            self.filenames = filenames[train_num:]
        self.size = len(self.filenames)
        
        self.transform = transform
        self.dims_x = data_config['feat_dims']

    def __getitem__(self, index):
        items = {"x": {}, "s": {}, "e": {}}
        
        for m, v in self.s_subset.items():
            items["s"][m] = np.array([v], dtype=np.int64)
        
        for m in self.comb:
            file_path = pj(self.in_dirs[m], self.filenames[index])
            v = np.array(utils.load_csv(file_path)[0])
            if m == "label":
                items["x"][m] = v.astype(np.int64)
            elif m == "atac":
                items["x"][m] = np.where(v.astype(np.float32) > 0.5, 1, 0).astype(np.float32)
            else:
                items["x"][m] = v.astype(np.float32)
                if self.transform is not None and m in self.transform.keys():
                    temp = np.zeros(self.dims_x[m], dtype=np.float32)
                    temp[self.transform[m][0]] = items["x"][m][self.transform[m][1]]
                    items["x"][m] = temp
            if m in self.masks.keys():
                items["e"][m] = self.masks[m]
                if self.transform is not None and m in self.transform.keys():
                    temp = np.zeros(self.dims_x[m], dtype=np.float32)
                    temp[self.transform[m][0]] = items["e"][m][self.transform[m][1]]
                    items["e"][m] = temp
        return items


    def __len__(self):
        return self.size

class MultiDatasetSampler2(th.utils.data.sampler.Sampler):

    def __init__(self, dataset, current_num=0, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.Sampler = th.utils.data.sampler.RandomSampler
        else:
            self.Sampler = th.utils.data.sampler.SequentialSampler
        self.number_of_datasets = len(dataset.datasets)
        # self.largest_dataset_size = max([cur_dataset.size for cur_dataset in dataset.datasets])
        # print(list(range(r, r+c)), [dataset.datasets[i].size for i in range(r, r+c)])
        self.r = len(dataset.datasets) - current_num
        self.c = current_num
        print('%d tasks used as replay data, %d tasks used as current training data' % (self.r, self.c))
        self.largest_dataset_size = max([dataset.datasets[i].size for i in range(self.r, self.r+self.c)]) # use the last dataset size to init sampler

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)
        # return max(self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets), self.r*2*self.batch_size)


    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = self.Sampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        self.samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets
        final_samples_list = []  # this is a list of indexes from the combined dataset
        n = 0
        m = 0
        # if (epoch_samples//step//2) >= self.r and self.r > 0:
        #     replay_list = [i for i in range(self.r)]
        # else:
        replay_list = [i for i in range(self.r)]
        random.shuffle(replay_list)
        current_list = [i for i in range(self.c)]
        random.shuffle(current_list)
        for _ in range(0, epoch_samples, step):
            # get the last dataset
            if self.c > 0:
                i = m % (self.c)
                cur_samples = self.get_sample_list(push_index_val[-1-current_list[i]], samplers_list[-1-current_list[i]], sampler_iterators[-1-current_list[i]])
                final_samples_list.extend(cur_samples)
                m += 1
            
            if self.r > 0:
                i = n % (self.number_of_datasets-self.c)
                cur_samples = self.get_sample_list(push_index_val[replay_list[i]], samplers_list[replay_list[i]], sampler_iterators[replay_list[i]])
                final_samples_list.extend(cur_samples)
                n += 1
                # get the rest datasets
            # for i in range(self.number_of_datasets-1):
        return iter(final_samples_list)
        
    def get_sample_list(self, push_index_val, samplers_list, sampler_iterators):
        cur_batch_sampler = sampler_iterators
        cur_samples = []
        for _ in range(self.samples_to_grab):
            try:
                cur_sample_org = cur_batch_sampler.__next__()
                cur_sample = cur_sample_org + push_index_val
                cur_samples.append(cur_sample)
            except StopIteration:
                # got to the end of iterator - restart the iterator and continue to get samples
                # until reaching "epoch_samples"
                sampler_iterators = samplers_list.__iter__()
                cur_batch_sampler = sampler_iterators
                cur_sample_org = cur_batch_sampler.__next__()
                cur_sample = cur_sample_org + push_index_val
                cur_samples.append(cur_sample)
        return cur_samples
    

class MultiDatasetSampler(th.utils.data.sampler.Sampler):

    def __init__(self, dataset, current_num=0, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.Sampler = th.utils.data.sampler.RandomSampler
        else:
            self.Sampler = th.utils.data.sampler.SequentialSampler
        self.number_of_datasets = len(dataset.datasets)
        # self.largest_dataset_size = max([cur_dataset.size for cur_dataset in dataset.datasets])
        # print(list(range(r, r+c)), [dataset.datasets[i].size for i in range(r, r+c)])
        self.r = len(dataset.datasets) - current_num
        self.c = current_num
        print('%d tasks used as replay data, %d tasks used as current training data' % (self.r, self.c))
        self.largest_dataset_size = max([dataset.datasets[i].size for i in range(self.r, self.r+self.c)]) # use the last dataset size to init sampler


    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = self.Sampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        self.samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        n = 0
        for _ in range(0, epoch_samples, step):
            # get the last dataset
            for t in range(self.c):
                cur_samples = self.get_sample_list(push_index_val[-1-t], samplers_list[-1-t], sampler_iterators[-1-t])
                final_samples_list.extend(cur_samples)
            
            if self.r > 0:
                i = n % (self.number_of_datasets-self.c)
                cur_samples = self.get_sample_list(push_index_val[i], samplers_list[i], sampler_iterators[i])
                final_samples_list.extend(cur_samples)
                n += 1
                # get the rest datasets

            # for i in range(self.number_of_datasets-1):

        return iter(final_samples_list)
        
    def get_sample_list(self, push_index_val, samplers_list, sampler_iterators):
        cur_batch_sampler = sampler_iterators
        cur_samples = []
        for _ in range(self.samples_to_grab):
            try:
                cur_sample_org = cur_batch_sampler.__next__()
                cur_sample = cur_sample_org + push_index_val
                cur_samples.append(cur_sample)
            except StopIteration:
                # got to the end of iterator - restart the iterator and continue to get samples
                # until reaching "epoch_samples"
                sampler_iterators = samplers_list.__iter__()
                cur_batch_sampler = sampler_iterators
                cur_sample_org = cur_batch_sampler.__next__()
                cur_sample = cur_sample_org + push_index_val
                cur_samples.append(cur_sample)
        return cur_samples


def get_dataset(subset=0, transform=True, subsample=True):
    if o.last_subsample_dir != '' and subset < len(o.s) - o.current_num and subsample==True:
        cell_id = list(np.array(utils.load_csv(pj(o.last_subsample_dir, 'subset_%d.csv'%subset)))[1:, 0].astype(np.int32)-1)
    else:
      cell_id = None
    dataset = MultimodalDataset(data_config, subset, o.transform[str(subset)] if transform else None, subsample=cell_id)
    print("Subset: %d, modalities %s, size: %d" %
        (subset, str(o.combs[subset]), dataset.size))
    return dataset

start = time.time()

data_config = None
net = None
discriminator = None
optimizer_net = None
optimizer_disc = None
benchmark = {
    "train_loss": [],
    "test_loss": [],
    "foscttm": [],
    "epoch_id_start": 0
}


combine_data()
# print(data_config['cell_names_path_ref'])

initialize()


if o.model_path != '' and o.init_model == '':
    update_model(o.model_path, net, discriminator)

if 'train' in o.actions or 'subsample' in o.actions or 'predict_subsample' in o.actions:
    datasets = []
    for i in range(len(o.s)):
        datasets.append(get_dataset(i))
    data_loader = get_dataloaders_cl(datasets, shuffle=True)


if 'train' in o.actions:
    print('training ... ')
    for epoch_id in range(benchmark['epoch_id_start'], o.epoch_num):
        run_epoch(data_loader, "train", epoch_id)
        check_to_save(epoch_id)

p_time_start = time.time()

if 'predict_subsample' in o.actions:
    o.pred_dir = pj(o.result_dir, 'predict_subsample')
    print('predicting with data involed in training ... ')
    predict(datasets, mod_latent=True)

datasets = []
for i in range(len(o.s)):
    datasets.append(get_dataset(i, subsample=False))
o.pred_dir = pj(o.result_dir, 'predict')

if 'predict_all' in o.actions:
    print('predicting with all data ... ')
    predict(datasets, mod_latent=True)
if "predict_joint" in o.actions:
    predict(datasets)
if "predict_all_latent" in o.actions:
    predict(datasets, mod_latent=True)
if "impute" in o.actions:
    predict(datasets, impute=True, input=True)
if "impute2" in o.actions:
    predict(datasets, impute=True)
if "translate" in o.actions:
    predict(datasets, translate=True, input=True)
if "batch_correct" in o.actions:
    predict(datasets, batch_correct=True, input=True)
if "predict_all_latent_bc" in o.actions:
    predict(datasets, mod_latent=True, batch_correct=True, input=True)

p_time_end = time.time()

if 'subsample' in o.actions:
    o.pred_dir = pj(o.result_dir, 'predict_subsample')
    print('subsampling, max storage size:%d' %o.max_size)
    subsample(o.pred_dir, size=o.max_size)

end  = time.time()

print('Total time: %f seconds'  %(end - start - (p_time_end - p_time_start)))
utils.save_list_to_csv([[end - start]], pj(o.result_dir, 'time.csv'))
utils.save_list_to_csv([[p_time_end - p_time_start]], pj(o.result_dir, 'p_time.csv'))
