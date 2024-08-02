import os
from os import path
from os.path import join as pj
import time
import argparse
from tqdm import tqdm
import math
import numpy as np
import torch as th
from torch import nn, autograd
import matplotlib.pyplot as plt
import re
import itertools
from modules import models, utils
from modules.datasets import MultimodalDataset_V2, MultiDatasetSampler
import pandas as pd
from sklearn.neighbors import BallTree
import random


class create_data():
    """
    An object named MIRACLE will be created to represent the data. The data information, including the cell number and features, will be automatically detected.
    """
    def __init__(self, data_path, mods=None, print_info=True):
        """
        Initialize a dataset 

        :param str data_path: Data path.
        :param dict mods: The dataset can have predefined modalities. If no modalities are given, all detected modalities will be used. 
        For example, you can specify the modalities as {"subset_0": ['rna', 'adt']}
        :param bool print_info: Whether to print the information. 
        """

        if mods is not None:
            self.mods = {k:utils.ref_sort(v, ref=['atac', 'rna', 'adt']) for k, v in mods.items()}
            self.predefine_mod = True
        else:
            self.predefine_mod = False
            self.mods = {}
        self.data_path = data_path
        self.__read_dir__()
        if print_info:
            self.info()
    
    def __read_dir__(self):
        """
        Read and check the data in the path.
        """

        assert os.path.exists(self.data_path), "This path does not exist."
        assert os.path.exists(os.path.join(self.data_path, 'feat')), "Feat dir does not exist."
        assert os.path.exists(os.path.join(self.data_path, 'feat', 'feat_dims.csv')), "Feat dimension 'feat_dims.csv' does not exist."
        
        self.subset = []
        self.cell_names = {}
        self.cell_names_orig = {}
        self.subset_cell_num = {}
        self.num_subset = 0
        self.features = {}
            
        for i in os.listdir(self.data_path):
            if 'subset_' in i:
                self.num_subset += 1
        for n in range(self.num_subset):
                i = 'subset_%d'%n
                self.subset.append(i)
                assert os.path.exists(os.path.join(self.data_path, i, 'cell_names.csv')), "'cell_names.csv' does not exist in %s."%i
                try:
                    self.cell_names[i] = pd.read_csv(os.path.join(self.data_path, i, 'cell_names_sampled.csv')).values[:, 1].flatten()
                except:
                    self.cell_names[i] = pd.read_csv(os.path.join(self.data_path, i, 'cell_names.csv')).values[:, 1].flatten()
                self.cell_names_orig[i] = pd.read_csv(os.path.join(self.data_path, i, 'cell_names.csv')).values[:, 1].flatten()
                self.subset_cell_num[i] = len(self.cell_names[i])
                
                if not self.predefine_mod:
                    m = []
                    for j in os.listdir(os.path.join(self.data_path, i, 'vec')):
                        if j in ['atac', 'rna', 'adt'] and os.path.exists(os.path.join(self.data_path, 'feat', 'feat_names_%s.csv'%j)):
                            m.append(j)
                    self.mods[i] = utils.ref_sort(m, ref=['atac', 'rna', 'adt'])

        self.mod_combination = utils.combine_mod(self.mods)

        for j in self.mod_combination:
            self.features[j] = pd.read_csv(os.path.join(self.data_path, 'feat', 'feat_names_%s.csv'%j), index_col=0).values.flatten().tolist()

        self.feat_dims = self.__cal_feat_dims__(pd.read_csv(os.path.join(self.data_path, 'feat', 'feat_dims.csv'), index_col=0), self.mod_combination)
        self.subset.sort()
        if 'atac' in self.mod_combination:
            self.dims_chr = pd.read_csv(os.path.join(self.data_path, 'feat', 'feat_dims.csv'), index_col=0)['atac'].values.tolist()
        else:
            self.dims_chr = []
    
    def __cal_feat_dims__(self, df, mods):
        """
        Calculate the feature dimensions.
        
        :param pandas.DataFrame df: Feature dimension for each modalities.
        :param list mods: Modalities.
        :return: dimensions of features.
        """
        feat_dims = {}
        # print(df)
        for c in df.columns:
            if c == 'atac' and c in mods:
                feat_dims['atac'] = df[c].values.sum().tolist()
            elif c in mods:
                feat_dims[c] = df[c].values[0].tolist()
        return feat_dims

    def info(self):
        """
        Print information.
        """
        print('%d subset(s) in this path' % self.num_subset, self.feat_dims)
        for key,value in self.subset_cell_num.items():
            print('%10s : %5d cells' % (key, value), ';', self.mods[key])


class MIRACLE():
    def __init__(self, data, status):
        """ 
        Initalize a miracle object.

        :param list data: A list of "create_data" object. The order of features is crucial as it determines their arrangement.
        :param list status: A list of status. "replay" or "current".
        """

        self.data = data
        self.batch_num_rep = 0
        self.batch_num_curr = 0
        self.mods = []
        for i, d in enumerate(data):
            if status[i] == 'replay':
                self.batch_num_rep += d.num_subset
            else:
                self.batch_num_curr += d.num_subset
            self.mods += list(d.mods.values())
        self.total_num = self.batch_num_rep + self.batch_num_curr
        self.s_joint = [[i] for i in range(self.total_num)]
        self.s_joint, self.combs, self.s, self.dims_s = utils.gen_all_batch_ids(self.s_joint, [self.mods])
        self.reference_features = {}
        self.dims_x = {}
        self.dims_chr = []
        self.dims_rep = {}
        for i, d in enumerate(data):
            # print(i)
            # d.info()
            for k in d.mod_combination:
                if k == 'atac':
                    self.dims_x['atac'] = d.feat_dims['atac']
                    self.reference_features[k] = d.features['atac']
                    self.dims_chr = d.dims_chr
                    if (status[i] == 'replay') and (k not in self.dims_rep):
                        self.dims_rep[k] = d.feat_dims[k]
                else:
                    if k not in self.reference_features:
                        self.reference_features[k] = d.features[k]
                    else:
                        self.reference_features[k], _ = utils.merge_features(self.reference_features[k].copy(), d.features[k].copy())
                    if status[i] == 'replay':
                        if k not in self.dims_rep:
                            self.dims_rep[k] = d.feat_dims[k]
                        else:
                            self.dims_rep[k]  = len(self.reference_features[k])
        for k in ['atac', 'rna', 'adt']:
            if k in self.reference_features:
                self.dims_x[k] = len(self.reference_features[k])
        
        self.mods = utils.ref_sort(np.unique(np.concatenate(self.mods).flatten()).tolist(), ['atac', 'rna','adt'])
        self.status = status
    def gen_datasets(self, data):

        """ 
        Generate dataset object.

        :param create_data object data: a "create_data" object.
        """

        datasets = []
        n = 0
        for d in data:
            for i in range(d.num_subset):
                datasets.append(MultimodalDataset_V2(d, subset=i, s_subset=self.s[n], reference_features=self.reference_features))
                n += 1
        return datasets
    
    def init_model(self, train_mod='denovo', lr=1e-4, drop_s=0, s_drop_rate=0.1, grad_clip=-1, dim_c=32, dim_b=2, dims_enc_s = [16,16], 
                    dims_enc_chr=[128,32], dims_enc_x=[1024,128], dims_discriminator=[128,64],norm="ln", drop=0.2, model_path=None, benchmark_path=None):
        """ 
        Initalize the model.

        :param str train_mod: The action to the model. 
                - denovo : To train a model without using replay data, 
                you can provide either the model_path or benchmark_path 
                parameter to load the weights or training status from a file. 
                If you wish to resume training from a specific checkpoint, 
                please ensure that you provide both the model_path and benchmark_path parameters.
                - continual : To train a model using replay data, make sure to initialize the object.
                with data under the "replay" status.
        :param float lr: Learning rate.
        :param int drop_s: Force to drop s.
        :param float s_drop_rate: Drop out rate for s.
        :param int grad_clip: Whether to clip the grad during training.
        :param str norm: Type of normalization.
        :param float drop: Drop out rate for the hidden layers.
        :param int dim_c: Dimension of the variable c.
        :param int dim_b: Dimension of the variable b.
        :param list dims_enc_s: Dimensions of the encoder layers for s.
        :param list dims_enc_chr: Dimensions of the encoder layers for chromosomes. (Used when there is atac data)
        :param list dims_enc_x: Dimensions of the encoder layers for data (except atac).
        :param list dims_discriminator: Dimensions of the discriminator layers.
        :param str model_path: The path for the model weight. A ".pt" file.
        :param str benchmark_path: The path for the training status. A ".toml" file.
        """

        assert not (train_mod != 'denovo' and model_path==None), 'Missing model weights path'
        dims_h = {}
        for m, dim in self.dims_x.items():
            dims_h[m] = dim if m != "atac" else dims_enc_chr[-1] * 22
        self.dims_h = dims_h
        self.benchmark = {
            "train_loss": [],
            "test_loss": [],
            "foscttm": [],
            "epoch_id_start": 0}

        self.o = utils.simple_obj({
            # data relevernt parameters
            'mods' : self.mods,
            'dims_x' : self.dims_x,
            'ref_mods': self.mods, # no meanings here
            's_joint' : self.s_joint,
            'combs' : self.combs, 
            's' : self.s, 
            'dims_s' : self.dims_s,
            'dims_chr' : self.dims_chr,
            # model hyper-parameters
            'drop' : drop,
            'drop_s' : drop_s,
            's_drop_rate' : s_drop_rate,
            'grad_clip' : grad_clip,
            'norm' : norm,
            'lr' : lr,
            # model structure
            'dim_c' : dim_c, 
            'dim_b' : dim_b, 
            'dim_s' : self.dims_s["joint"], 
            'dim_z' : dim_c + dim_b, 
            'dims_enc_s' : dims_enc_s, 
            'dims_enc_chr' : dims_enc_chr, 
            'dims_enc_x' : dims_enc_x, 
            'dims_dec_x' : dims_enc_x[::-1],
            'dims_dec_s' : dims_enc_s[::-1],
            'dims_dec_chr' : dims_enc_chr[::-1],
            "dims_h" : dims_h,
            'dims_discriminator' : dims_discriminator
            })

        self.net = models.Net(self.o).cuda()
        self.discriminator = models.Discriminator(self.o).cuda()
        self.optimizer_net = th.optim.AdamW(self.net.parameters(), lr=self.o.lr)
        self.optimizer_disc = th.optim.AdamW(self.discriminator.parameters(), lr=self.o.lr)

        if train_mod == 'scArches':
            print('load an old model from', model_path)
            savepoint = th.load(model_path)
            self.past_savepoint = savepoint
            dims_h_rep = {}
            for m, dim in self.dims_rep.items():
                dims_h_rep[m] = dim if m != "atac" else dims_enc_chr[-1] * 22
            self.dims_h_rep = dims_h_rep
            self.net = update_model(savepoint, dims_h_rep, self.o.dims_h, self.net)
            self.discriminator = models.Discriminator(self.o).cuda()
            self.discriminator = update_discriminator(savepoint, self.discriminator)
            self.optimizer_net = th.optim.AdamW(self.net.parameters(), lr=self.o.lr)
            self.optimizer_disc = th.optim.AdamW(self.discriminator.parameters(), lr=self.o.lr)
            temp  = []
            for i, j in enumerate(self.status):
                if j=='current':
                    temp.append(self.data[i])
            self.data = temp
            self.s = self.s[self.batch_num_rep:]
            # self.data = self.data[self.status=='current']
            # if type(self.data) != list:
            #     self.data = [self.data]
            # print(self.data)
            self.use_rnt = False
        else:
            if model_path is not None:
                print('load a pretrained model from', model_path)
                savepoint = th.load(model_path)
                self.net.load_state_dict(savepoint['net_states'])
                self.discriminator.load_state_dict(savepoint['disc_states'])
                self.optimizer_net.load_state_dict(savepoint['optim_net_states'])
                self.optimizer_disc.load_state_dict(savepoint['optim_disc_states'])
            if benchmark_path is not None:
                savepoint_toml = utils.load_toml(benchmark_path)
                self.benchmark.update(savepoint_toml['benchmark'])

        net_param_num = sum([param.data.numel() for param in self.net.parameters()])
        disc_param_num = sum([param.data.numel() for param in self.discriminator.parameters()])
        print('Parameter number: %.3f M' % ((net_param_num+disc_param_num) / 1e6))

    def __forward_net__(self, inputs):
        return self.net(inputs)


    def __forward_disc__(self, c, s):
        return self.discriminator(c, s)
    
    # def __update_disc__(self, loss):
    #     self.__update__(loss, self.discriminator, self.optimizer_disc)


    def __update_net__(self,loss):
        self.optimizer_net.zero_grad()
        loss.backward()
        if self.o.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.o.grad_clip)
        # key step for scArches
        # freeze_model(self.past_savepoint, self.dims_h_rep, self.dims_h, self.net)
        self.optimizer_net.step()

    def __update_disc__(self,loss):
        self.optimizer_disc.zero_grad()
        loss.backward()
        if self.o.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.o.grad_clip)
        # key step for scArches
        # freeze_disc(self.past_savepoint, self.discriminator)
        self.optimizer_disc.step()
        
    def __run_iter__(self, split, epoch_id, inputs, rnt=1):
        inputs = utils.convert_tensors_to_cuda(inputs)
        if split == "train":
            with autograd.set_detect_anomaly(self.debug == 1):
                loss_net, c_all = self.__forward_net__(inputs)
                self.discriminator.epoch = epoch_id
                K = 3
                for _ in range(K):
                    loss_disc = self.__forward_disc__(utils.detach_tensors(c_all), inputs["s"])
                    loss_disc = loss_disc * rnt
                    self.__update_disc__(loss_disc)
                loss_adv = self.__forward_disc__(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
                loss = rnt * loss
                self.__update_net__(loss)
            
        else:
            with th.no_grad():
                loss_net, c_all = self.__forward_net__(inputs)
                loss_adv = self.__forward_disc__(c_all, inputs["s"])
                loss_adv = -loss_adv
                loss = loss_net + loss_adv
        return loss.item()

    def __run_epoch__(self, data_loader, split, epoch_id=0):
        start_time = time.time()
        if split == "train":
            self.net.train()
            self.discriminator.train()
        elif split == "test":
            self.net.eval()
            self.discriminator.eval()
        else:
            assert False, "Invalid split: %s" % split
        loss_total = 0
        for i, data in enumerate(data_loader):
            # print(data['s']['joint'][0])
            if self.batch_num_rep == 0:
                rnt_ = 1
            elif self.batch_num_rep == 0 or i%2 < 1:
                rnt_ = self.batch_num_curr / (self.batch_num_rep * self.batch_num_rep + self.batch_num_curr * self.batch_num_curr)
            else:
                rnt_ = self.batch_num_rep / (self.batch_num_rep * self.batch_num_rep + self.batch_num_curr * self.batch_num_curr)
            if not self.use_rnt:
                loss = self.__run_iter__(split, epoch_id, data, rnt_)
            else:
                loss = self.__run_iter__(split, epoch_id, data, 1)
            loss_total += loss
        loss_avg = loss_total / len(data_loader)
        epoch_time = (time.time() - start_time) / 3600 / 24
        # elapsed_time = epoch_time * (epoch_id+1)
        # total_time = epoch_time * o.epoch_num
        self.benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
        return loss_avg, epoch_time

    def train(self, n_epoch=2000, mini_batch_size=256, log_epochs=100, shuffle=True, save_epochs=10, debug=0, save_path='./result/experiment/'):
        print("Training ...")
        """ 
        Start training the model.
        
        :param int n_epoch: Training epoch. 
        :param int mini_batch_size: Mini-batch size.
        :param bool shuffle: Whether to train the model with data out of order.
        :param int save_epochs: An integer. Epochs to save the latest weights.
        :param int log_epochs: An integer. Epochs to save the training status (overwrite previous ones).
        :param int debug: An integer. Print intermediate variables.
        :param str save_path: A string. Save files to this path.
        """

        self.save_epochs = save_epochs
        self.log_epochs = log_epochs
        self.debug = debug
        self.o.debug = debug
        # print(self.data)
        datasets = self.gen_datasets(self.data)
        # print(datasets)
        sampler = MultiDatasetSampler(th.utils.data.dataset.ConcatDataset(datasets), batch_size=mini_batch_size, shuffle=shuffle)
        self.data_loader = th.utils.data.DataLoader(th.utils.data.dataset.ConcatDataset(datasets), batch_size=mini_batch_size, sampler=sampler, num_workers=64, pin_memory=True)
        # loop = tqdm.tqdm(total=n_epoch)
        with tqdm(total=n_epoch) as pbar:
            pbar.update(self.benchmark['epoch_id_start'])
            for epoch_id in range(self.benchmark['epoch_id_start'], n_epoch):
                loss_avg, epoch_time = self.__run_epoch__(self.data_loader, "train", epoch_id)
                # print(epoch_id,loss_avg )
                self.__check_to_save__(epoch_id, n_epoch, save_path)
                pbar.update(1)
                pbar.set_description("Loss: %.4f" % loss_avg)
        
    def predict(self, save_dir='./result/experiment/predict/', joint_latent=True, mod_latent=False, impute=False, batch_correct=False, translate=False, 
            input=False, mini_batch_size=256):
        """ 
        Predict the embeddings or their imputed expression.

        :param str save_dir: The path to save the predicted files.
        :param bool joint_latent: Whether to save the joint embeddings.
        :param bool mod_latent: Whether to save the embeddings for each modalities.
        :param bool impute: Whether to save the imputed data.
        :param bool batch_correct: Whether to save the batch corrected expression data.
        :param bool translate: Whether to save the translation embeddings (from a modality to another modality).
        :param bool input: Whether to save the inputs.
        :param int mini_batch_size: The mini-batch size for saving. Influence the cell number in the csv file.
        """

        if translate:
            mod_latent = True
        print("Predicting ...")
        self.o.pred_dir = save_dir
        if not os.path.exists(self.o.pred_dir):
            os.makedirs(self.o.pred_dir)
        dirs = utils.get_pred_dirs(self.o, joint_latent, mod_latent, impute, batch_correct, translate, input)
        parent_dirs = list(set(map(path.dirname, utils.extract_values(dirs))))
        utils.mkdirs(parent_dirs, remove_old=True)
        utils.mkdirs(dirs, remove_old=True)
        datasets = self.gen_datasets(self.data)
        data_loaders = {k:th.utils.data.DataLoader(datasets[k], batch_size=mini_batch_size, \
            num_workers=64, pin_memory=True, shuffle=False) for k in range(self.batch_num_curr+self.batch_num_rep)}
        # data_loaders = get_dataloaders("test", train_ratio=0)
        self.net.eval()
        with th.no_grad():
            for subset_id, data_loader in data_loaders.items():
                print("Processing subset %d: %s" % (subset_id, str(self.o.combs[subset_id])))
                fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                
                for i, data in enumerate(tqdm(data_loader)):
                    data = utils.convert_tensors_to_cuda(data)
                    
                    # conditioned on all observed modalities
                    if joint_latent:
                        x_r_pre, _, _, _, z, _, _, *_ = self.net.sct(data)  # N * K
                        utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["joint"], fname_fmt) % i)
                    if impute:
                        x_r = models.gen_real_data(x_r_pre, sampling=False)
                        for m in self.o.mods:
                            utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_impt"][m], fname_fmt) % i)
                    if input:  # save the input
                        for m in self.o.combs[subset_id]:
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
                            x_r_pre, _, _, _, z, c, b, *_ = self.net.sct(input_data)  # N * K
                            utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"][m], fname_fmt) % i)
                            if translate: # single to double
                                x_r = models.gen_real_data(x_r_pre, sampling=False)
                                for m_ in set(self.o.mods) - {m}:
                                    utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m+"_to_"+m_], fname_fmt) % i)
                    
                    if translate: # double to single
                        for mods in itertools.combinations(data["x"].keys(), 2):
                            m1, m2 = utils.ref_sort(mods, ref=self.o.mods)
                            input_data = {
                                "x": {m1: data["x"][m1], m2: data["x"][m2]},
                                "s": data["s"], 
                                "e": {}
                            }
                            for m in mods:
                                if m in data["e"].keys():
                                    input_data["e"][m] = data["e"][m]
                            x_r_pre, *_ = self.net.sct(input_data)  # N * K
                            x_r = models.gen_real_data(x_r_pre, sampling=False)
                            m_ = list(set(self.o.mods) - set(mods))[0]
                            utils.save_tensor_to_csv(x_r[m_], pj(dirs[subset_id]["x_trans"][m1+"_"+m2+"_to_"+m_], fname_fmt) % i)

            if batch_correct:
                print("Calculating b_centroid ...")
                # z, c, b, subset_ids, batch_ids = utils.load_predicted(o)
                # b = th.from_numpy(b["joint"])
                # subset_ids = th.from_numpy(subset_ids["joint"])
                
                pred = utils.load_predicted(self.o)
                b = th.from_numpy(pred["z"]["joint"][:, self.o.dim_c:])
                s = th.from_numpy(pred["s"]["joint"])

                b_mean = b.mean(dim=0, keepdim=True)
                b_subset_mean_list = []
                for subset_id in s.unique():
                    b_subset = b[s == subset_id, :]
                    b_subset_mean_list.append(b_subset.mean(dim=0))
                b_subset_mean_stack = th.stack(b_subset_mean_list, dim=0)
                dist = ((b_subset_mean_stack - b_mean) ** 2).sum(dim=1)
                self.net.sct.b_centroid = b_subset_mean_list[dist.argmin()]
                self.net.sct.batch_correction = True
                
                print("Batch correction ...")
                for subset_id, data_loader in data_loaders.items():
                    print("Processing subset %d: %s" % (subset_id, str(self.o.combs[subset_id])))
                    fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
                    
                    for i, data in enumerate(tqdm(data_loader)):
                        data = utils.convert_tensors_to_cuda(data)
                        x_r_pre, *_ = self.net.sct(data)
                        x_r = models.gen_real_data(x_r_pre, sampling=True)
                        for m in self.o.mods:
                            utils.save_tensor_to_csv(x_r[m], pj(dirs[subset_id]["x_bc"][m], fname_fmt) % i)
    
    def read_embeddings(self, emb_path=None, joint_latent=True, mod_latent=False, impute=False, batch_correct=False, 
                   translate=False, input=False, group_by="modality"):
        """ 
        Get embeddings or other outputs from a path.

        :param str emb_path: If not given, use the path from the function predict() if you have run it.
        :param bool joint_latent: Whether to read the joint embeddings.
        :param bool impute: Whether to read the imputed expression data.
        :param bool batch_correct: Whether to read the batch corrected expression data.
        :param bool translate: Whether to read the translated embeddings.
        :param bool input: Whether to read the inputs.
        :param str group_by: Group the data by "modality" or "batch".
        :return: The embeddings from the given path.
        """

        if emb_path is not None:
            self.o.pred_dir = emb_path
        pred = utils.load_predicted(self.o, joint_latent=True, mod_latent=False, impute=False, batch_correct=False, 
                   translate=False, input=False, group_by="modality")
        return pred

    def BallTreeSubsample(self, X, target_size, ls=2):
        """ 
        Sample using the ball-tree.

        :param numpy.array X: A matrix (sample * feature).
        :param int target: An integer for the target number.
        :param int ls: The leaf size of the ball-tree.
        :return: ID of samples.
        """

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

    def pack(self, output_task_name='pack', des_dir='./data/processed/', n_sample=100000, pred_dir=None):
        """ 
        Pack the data for future training or sharing.

        :param str output_task_name: The output name. des_dir + output_task_name
        :param str des_dir: Path to save the data.
        :param int n_sample: The target number of samples.
        :param str pred_dir: Path of the embeddings.
        """

        if pred_dir is None:
            pred_dir = self.o.pred_dir
        else:
            self.o.pred_dir = pred_dir
        print("Packing ...")

        if not os.path.exists(pj(des_dir, output_task_name)):
            os.makedirs(pj(des_dir, output_task_name, 'feat'))

        # load info
        datasets = self.gen_datasets(self.data)
        data_loaders = {k:th.utils.data.DataLoader(datasets[k], batch_size=1, \
            num_workers=64, pin_memory=True, shuffle=False) for k in range(self.batch_num_curr+self.batch_num_rep)}
        
        emb = self.read_embeddings()
        # print(emb)
        cell_names = {}
        cell_names_sampled = {}
        cell_nums = []
        n = 0
        for d in self.data:
            for k in range(d.num_subset):
                cell_names_sampled[n] = d.cell_names['subset_%d'%k]
                cell_names[n] = d.cell_names_orig['subset_%d'%k]
                cell_nums.append(len(cell_names[n]))
                n += 1

        # cell_nums = np.concatenate([d.subset_cell_num.values for d in self.data]).flatten().tolist()

        # for i in range(self.batch_num_rep):
            # cell_names[i] = pd.read_csv(pj(self.data_path, self.replay_task, 'subset_%d'%i, 'cell_names.csv'), index_col=0)
            # cell_nums.append(len(cell_names[i]))
            # if os.path.exists(pj(self.data_path, self.replay_task, 'subset_%d'%i, 'cell_names_sampled.csv')):
                # cell_names_sampled[i] = pd.read_csv(pj(self.data_path, self.replay_task, 'subset_%d'%i, 'cell_names_sampled.csv'), index_col=0)
        # for i in range(self.batch_num_curr):
            # cell_names[i+self.batch_num_rep] = pd.read_csv(pj(self.data_path, self.current_task, 'subset_%d'%i, 'cell_names.csv'), index_col=0)
            # cell_nums.append(len(cell_names[i+self.batch_num_rep]))
            # if os.path.exists(pj(self.data_path, self.current_task, 'subset_%d'%i, 'cell_names_sampled.csv')):
                # cell_names_sampled[i] = pd.read_csv(pj(self.data_path, self.current_task, 'subset_%d'%i, 'cell_names_sampled.csv'), index_col=0)

        if sum(cell_nums) > n_sample:
            rate = (np.array(cell_nums) / sum(cell_nums)  * n_sample).astype(int)
        else:
            rate = [len(i) for i in datasets]
        sample_preserve = {}
        if not os.path.exists(pj(des_dir, output_task_name, 'feat')):
            os.makedirs(pj(des_dir, output_task_name, 'feat'))
        for i in range(self.batch_num_rep+self.batch_num_curr):
            emb_subset = emb['z']['joint'][emb['s']['joint']==i]
            sample_preserve['subset_%d'%i] = self.BallTreeSubsample(emb_subset, rate[i])
            # print(sample_preserve['subset_%d'%i])
            sample_preserve['subset_%d'%i].sort()
            # print(sample_preserve['subset_%d'%i])

            if not os.path.exists(pj(des_dir, output_task_name, 'subset_%d'%i, 'mask')):
                os.makedirs(pj(des_dir, output_task_name, 'subset_%d'%i, 'mask'))
            if i in cell_names_sampled:
                pd.DataFrame(cell_names_sampled[i][sample_preserve['subset_%d'%i]]).to_csv(pj(des_dir, output_task_name, 'subset_%d'%i, 'cell_names_sampled.csv'))
                pd.DataFrame(cell_names[i]).to_csv(pj(des_dir, output_task_name, 'subset_%d'%i, 'cell_names.csv'))
            else:
                pd.DataFrame(cell_names[i][sample_preserve['subset_%d'%i]]).to_csv(pj(des_dir, output_task_name, 'subset_%d'%i, 'cell_names_sampled.csv'))
                pd.DataFrame(cell_names[i]).to_csv(pj(des_dir, output_task_name, 'subset_%d'%i, 'cell_names.csv'))
            fname_fmt = utils.get_name_fmt(rate[i])+".csv"
            n = 0
            for k, data in enumerate(data_loaders[i]):
                if k in sample_preserve['subset_%d'%i]:
                    for m in self.o.combs[i]:
                        if not os.path.exists(pj(des_dir, output_task_name, 'subset_%d'%i, 'vec', m)):
                            os.makedirs(pj(des_dir, output_task_name, 'subset_%d'%i, 'vec', m))
                        utils.save_tensor_to_csv(data["x"][m].int(), pj(des_dir, output_task_name, 'subset_%d'%i, 'vec', m, fname_fmt) % n)
                    n += 1
            for k, data in enumerate(data_loaders[i]):
                for m in self.o.combs[i]:
                    if m != 'atac':
                        pd.DataFrame(utils.convert_tensor_to_list(data["e"][m].int())).to_csv(pj(des_dir, output_task_name, 'subset_%d'%i, 'mask', '%s.csv'%m))
                break

        features_dims = {}
        if self.dims_chr != []:
            features_dims['atac'] = self.dims_chr
        for m in self.reference_features:
            if m != 'atac':
                features_dims[m] = [self.dims_x[m] for i in range(22)]
            pd.DataFrame(self.reference_features[m]).to_csv(pj(des_dir, output_task_name, 'feat','feat_names_%s.csv'%m))
        pd.DataFrame(features_dims).to_csv(pj(des_dir, output_task_name, 'feat','feat_dims.csv'))

    def viz_loss(self):
        """ 
        Visualize the loss
        """
        plt.figure(figsize=(4,2))
        plt.plot(np.array(self.benchmark['train_loss'])[:, 0]+1, np.array(self.benchmark['train_loss'])[:, 1])
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')
        plt.title('Loss curve')

    def __check_to_save__(self, epoch_id, epoch_num, save_path):
        if (epoch_id+1) % self.log_epochs == 0 or epoch_id+1 == epoch_num:
            self.__save_training_states__(epoch_id, "sp_%08d" % epoch_id, save_path)
        if (epoch_id+1) % self.save_epochs == 0 or epoch_id+1 == epoch_num:
            self.__save_training_states__(epoch_id, "sp_latest", save_path)
    
    def __save_training_states__(self, epoch_id, filename,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.benchmark['epoch_id_start'] = epoch_id + 1
        utils.save_toml({"o": vars(self.o), "benchmark": self.benchmark}, pj(save_path, filename+".toml"))
        th.save({"net_states": self.net.state_dict(),
                "disc_states": self.discriminator.state_dict(),
                "optim_net_states": self.optimizer_net.state_dict(),
                "optim_disc_states": self.optimizer_disc.state_dict()
                }, pj(save_path, filename+".pt"))
        
# def freeze_model(savepoint, dims_h_past, dims_h, curr_model):
#     past_model = savepoint['net_states']
#     for p in curr_model.named_parameters():
#         k = p[0]
#         v = p[1]
#     # for k, v in curr_model.items():
#         if k not in past_model.keys():
#             pass
#         elif k == 'sct.x_dec.net.4.weight' or k == 'sct.x_dec.net.4.bias':
#             param_dict_last = dict(zip(dims_h_past.keys(), past_model[k].split(list(dims_h_past.values()), dim=0)))
#             param_dict_len = [i.shape[0] for i in v.split(list(dims_h.values()), dim=0)]
#             shape_list = dict(zip(dims_h.keys(), param_dict_len))
#             start_id = 0
#             for m in dims_h_past.keys():
#                 v.grad[start_id:start_id+param_dict_last[m].shape[0]] = th.zeros_like(param_dict_last[m])
#                 start_id += shape_list[m]
#         elif v.shape == past_model[k].shape:
#             v.grad = past_model[k]
#         elif len(v.shape)==2 and v.shape[0] == past_model[k].shape[0]:
#             v.grad[:, :past_model[k].shape[1]] = th.zeros_like(past_model[k])
#         elif len(v.shape)==2 and v.shape[1] == past_model[k].shape[1]:
#             v.grad[:past_model[k].shape[0], :] = th.zeros_like(past_model[k])
#         else:
#             v.grad[:past_model[k].shape[0]] = th.zeros_like(past_model[k])

# def freeze_disc(savepoint, disc):
#     past_model = savepoint['disc_states']
#     for p in disc.named_parameters():
#         k = p[0]
#         v = p[1]
#     # for k, v in temp_model.items():
#         if k not in past_model.keys():
#             pass
#         elif v.shape == past_model[k].shape:
#             v.grad = th.zeros_like(past_model[k])
#         elif len(v.shape)==2 and v.shape[0] == past_model[k].shape[0]:
#             # print('padding')
#             v.grad[:, :past_model[k].shape[1]] = th.zeros_like(past_model[k])

def update_discriminator(savepoint, discriminator):
    past_model = savepoint['disc_states']
    temp_model = discriminator.state_dict()
    for k, v in temp_model.items():
        if k not in past_model.keys():
            pass
        elif v.shape == past_model[k].shape:
            past_model[k].requires_grad = False
            temp_model[k] = past_model[k]
        elif len(v.shape)==2 and v.shape[0] == past_model[k].shape[0]:
            # print('padding')
            past_model[k].requires_grad = False
            temp_model[k][:, :past_model[k].shape[1]] = past_model[k]
    discriminator.load_state_dict(temp_model)
    return discriminator

def update_model(savepoint, dims_h_past, dims_h, curr_model):
    past_model = savepoint['net_states']
    temp_model = curr_model.state_dict()
    for k, v in temp_model.items():
        # print(k, v.shape)
        if k not in past_model.keys():
            pass
            # print('not in new, pass')
        elif k == 'sct.x_dec.net.4.weight' or k == 'sct.x_dec.net.4.bias':
            # print('last layer')
            param_dict_last = dict(zip(dims_h_past.keys(), past_model[k].split(list(dims_h_past.values()), dim=0)))
            param_dict_len = [i.shape[0] for i in temp_model[k].split(list(dims_h.values()), dim=0)]
            shape_list = dict(zip(dims_h.keys(), param_dict_len))
            start_id = 0
            for m in dims_h_past.keys():
                param_dict_last[m].requires_grad = False
                temp_model[k][start_id:start_id+param_dict_last[m].shape[0]] = param_dict_last[m]
                start_id += shape_list[m]
        elif v.shape == past_model[k].shape:
            # print('same shape layer')
            past_model[k].requires_grad = False
            temp_model[k] = past_model[k]
        elif len(v.shape)==2 and v.shape[0] == past_model[k].shape[0]:
            # print('padding')
            past_model[k].requires_grad = False
            temp_model[k][:, :past_model[k].shape[1]] = past_model[k]
        elif len(v.shape)==2 and v.shape[1] == past_model[k].shape[1]:
            # print('padding')
            temp_model[k][:past_model[k].shape[0], :] = past_model[k]
        else:
            # print('padding')
            past_model[k].requires_grad = False
            temp_model[k][:past_model[k].shape[0]] = past_model[k]
        curr_model.load_state_dict(temp_model)
    return curr_model