# %%
import os
from os.path import join as pj
import argparse
from modules import utils
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='',
    help="Choose a continual task")
parser.add_argument('--exp_prefix', type=str, default='c_',
    help="Define a uniform name for every step of experiments")
parser.add_argument('--actions', type=str, nargs='+', default=['train', 'subsample', 'predict_subsample', 'predict_all'],
    help="Choose actions to run. predict_subsample is used for subsampling only. predict_all is used to predict all data")
parser.add_argument('--max_size', type=int, default=100000,
    help="The max storage size of samples.")
parser.add_argument('--start', type=int, default=0,
    help="Step you want to start training.")
parser.add_argument('--end', type=int, default=100,
    help="Step you want to end training.")
parser.add_argument('--cuda', type=int, default=1,
    help="Define gpu.")
parser.add_argument('--init_model', type=str, default="",
    help="Load model from this file under path /train ")
parser.add_argument('--epoch_list', type=str, nargs='+', default=['2000'],
    help="The epoch num for every steps.")
parser.add_argument('--use_predefined_mods', type=int, default=1,
    help="Use predefine mods in continual task")
parser.add_argument('--REPLAY', type=int, default=1,
    help="")
parser.add_argument('--subsample_method', type=str, choices=['bts', 'random'], default='bts',
    help="")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
parser.add_argument('--denovo', type=int, default=1,
    help="Choose initial actions")
parser.add_argument('--use_shm', type=int, default=1,
    help="Use shared memory to accelerate training")
parser.add_argument('--seed', type=int, default=42,
    help="Set the random seed to reproduce the results")
parser.add_argument('--adv', type=int, default='30',
    help="choose the resistance coefficient")
o = parser.parse_args()
# %%

data_config = utils.load_toml("configs/data.toml")[o.task]

# %%
data_current = []
data_replay = []

# %%
tasks = data_config['tasks']   # The config of continual task has an extra item [tasks] which defines the tasks to train step by step.
end = min(len(tasks), o.end+1)

# %%
for i in range(o.start, end):

    if len(o.epoch_list) < i+1:
        o.epoch_num = o.epoch_list[-1]
    else:
        o.epoch_num = o.epoch_list[i]

    # o.experiment = o.exp_prefix+str(i)+'_'+'ep'+o.epoch_num
    # o.last_experiment = o.exp_prefix+str(i-1)+'_'+'ep'+o.epoch_num
    o.experiment = o.exp_prefix+str(i)
    o.last_experiment = o.exp_prefix+str(i-1)

    data_current = [tasks[i]]
    data_replay = " ".join(tasks[:i])  if i > 0 else None
    if o.denovo == 0:
        model_path = pj('result', o.task, o.last_experiment, o.model, 'train', 'sp_latest.pt') if i > 0 else pj('result', o.task, o.exp_prefix+str(0), o.model, 'train', 'sp_latest.pt')
        initial_actions = 'subsample predict_subsample'
    elif o.denovo == 1:
        model_path = pj('result', o.task, o.last_experiment, o.model, 'train', 'sp_latest.pt') if i > 0 else "''"
        initial_actions = " ".join(o.actions)
    else:
        print('Unknown denovo parameters!')
    last_subsample_dir = pj('result', o.task, o.last_experiment, o.model, 'subsample') if i > 0 else "''"
    print('-'*20,'Step: %d' % i, '-'*20)
    print('CUDA : %d \nTASK : %s \nEXPERIMENT : %s \nEPOCH NUM : %s \nDATA CURRENT : %s \nDATA REPLAY: %s \nUSE REPLAY : %d \nMODEL PATH : %s \nMAX STORAGE : %d \nSUBSAMPLE MOTHED : %s \nLAST SUBSAMPLE DIR : %s \nACTIONS : %s \nDENOVO : %d \nUSE SHM : %d \nSEED : %d \nADV : %d' \
          % (o.cuda, o.task, o.experiment, o.epoch_num, data_current[0], data_replay, o.REPLAY, model_path, o.max_size, o.subsample_method, last_subsample_dir, " ".join(o.actions), o.denovo, o.use_shm, o.seed, o.adv))
    
    if data_replay is not None:
        os.system('CUDA_VISIBLE_DEVICES=%d python continual.py --task %s --experiment %s --data_current %s --data_replay %s --model_path %s \
            --last_subsample_dir %s --max_size %d --epoch_num %s --actions %s --init_model "%s" --use_predefined_mods %s --REPLAY %d --subsample_method %s --model %s --use_shm %d --seed %d --adv %d' \
            % (o.cuda, o.task, o.experiment, data_current[0], data_replay, model_path, last_subsample_dir, o.max_size, \
                o.epoch_num, " ".join(o.actions), o.init_model, o.use_predefined_mods, o.REPLAY, o.subsample_method, o.model, o.use_shm, o.seed, o.adv))
    else:
        # pass
        os.system('CUDA_VISIBLE_DEVICES=%d python continual.py --task %s --experiment %s --data_current %s  --model_path %s \
            --last_subsample_dir %s --max_size %d --epoch_num %s --actions %s --init_model "%s" --use_predefined_mods %s --REPLAY %d --subsample_method %s --model %s --use_shm %d --seed %d --adv %d' \
            % (o.cuda, o.task, o.experiment, data_current[0], model_path, last_subsample_dir, o.max_size, o.epoch_num, initial_actions, o.init_model, o.use_predefined_mods, o.REPLAY, o.subsample_method, o.model, o.use_shm, o.seed, o.adv))