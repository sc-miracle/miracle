## Configure the data

Let's add the following items to the file `./config/data.toml`.

```
[dig_ctrl]
raw_data_dirs = ["data/raw/atac+rna+adt/dogma/dig_ctrl"]
raw_data_frags = ["GSM5065530_DIG_CTRL_fragments.tsv.gz"]
combs = [[ ["atac", "adt"] ]]
comb_ratios = [[ 1 ]]
s_joint = [[ 0 ]]
train_ratio = 1
N = 256

[dig_stim]
raw_data_dirs = ["data/raw/atac+rna+adt/dogma/dig_stim"]
raw_data_frags = ["GSM5065533_DIG_STIM_fragments.tsv.gz"]
combs = [[ ["atac", "rna"] ]]
comb_ratios = [[ 1 ]]
s_joint = [[ 0 ]]
train_ratio = 1
N = 256

[lll_ctrl]
raw_data_dirs = ["data/raw/atac+rna+adt/dogma/lll_ctrl"]
raw_data_frags = ["GSM5065530_DIG_CTRL_fragments.tsv.gz"]
combs = [[ ["rna", "adt"] ]]
comb_ratios = [[ 1 ]]
s_joint = [[ 0 ]]
train_ratio = 1
N = 256

[lll_stim]
raw_data_dirs = ["data/raw/atac+rna+adt/dogma/lll_stim"]
raw_data_frags = ["GSM5065527_LLL_STIM_fragments.tsv.gz"]
combs = [[ ["atac", "rna", "adt"] ]]
comb_ratios = [[ 1 ]]
s_joint = [[ 0 ]]
train_ratio = 1
N = 256
```
