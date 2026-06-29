# MIRACLE: Continual Integration for Single-Cell Multimodal Data

MIRACLE is a Python framework for scalable continual integration of single-cell
multimodal datasets. It builds on the MIDAS modeling stack and adds
MuData-native utilities for sequential data releases, replay-aware training,
feature-space expansion, checkpoint transfer, and latent representation
extraction. MIRACLE is designed for scenarios where new batches, modalities, or
features arrive over time and should be incorporated without retraining the full
historical dataset from scratch.

## Reproducibility

Code and notebooks used to reproduce the experiments in the
MIRACLE paper are provided in the `reproducibility/` directory.

## Documentation

Full tutorials and API documentation are available at:

https://miracle-docs.readthedocs.io/en/latest/

## Installation

MIRACLE requires Python 3.10 or later. Python 3.11 is recommended.

```bash
conda create -n scmiracle python=3.11
conda activate scmiracle
pip install scmiracle==0.1.0
```

## Data

MIRACLE provides a small downloader for prepared MuData archives used by the
tutorials.

```python
from scmiracle.download_data import download, list_available_datasets

list_available_datasets()
download("DOTEA_mudata", "./data")
download("atlas_transfer_mudata", "./data")
```

## Basic Usage

MIRACLE works with `mudata.MuData` objects. Each modality should be stored in
`mdata.mod`, and batch labels should be available under the observation key
`batch` by default.

### 1. De Novo Training

Use de novo training for the first task or for a standalone integration run.

```python
import mudata as mu

from scmiracle.model import MIRACLE

mdata = mu.read_h5mu("./data/DOTEA_mudata/step1.h5mu")

miracle = MIRACLE(
    batch_key="batch",
    batch_size=128,
    save_model_path="./saved_models/miracle_step1",
)

miracle.setup_mudata(mdata)
miracle.train(max_epochs=2000, accelerator="auto", devices=1)

mdata.obsm["X_miracle"] = miracle.get_latent_representation(kind="joint")
model_dir = miracle.save()
```

### 2. Compress Data for Replay

After training, compress the source data into a replay memory. The replay data
keeps representative cells in the learned latent space and stores metadata
needed by future continual training steps.

```python
replay = miracle.build_replay(
    replay_size=2000,
    source_mdata=mdata,
    latent=mdata.obsm["X_miracle"],
    strategy="subsample",
)

replay.write_h5mu("./saved_models/replay_step1.h5mu")
MIRACLE.export_mudata_metadata(
    replay,
    "./saved_models/replay_step1.json",
    replay_strategy="subsample",
)
```

Use `strategy="full"` for small datasets when you want to keep every cell.

### 3. Continual Training

For a new data release, load the previous replay memory and checkpoint, then
train on the replay data plus the current MuData object.

```python
current = mu.read_h5mu("./data/DOTEA_mudata/step2.h5mu")
replay = MIRACLE.load_replay_with_metadata(
    "./saved_models/replay_step1.h5mu",
    "./saved_models/replay_step1.json",
)

miracle = MIRACLE(
    batch_key="batch",
    batch_size=128,
    save_model_path="./saved_models/miracle_step2",
)

miracle.setup_continual(
    current,
    replay_mdata=replay,
    prev_model_dir=model_dir,
    lazy=True,
    inherit_dsc=True,
)
miracle.train(max_epochs=2000, accelerator="auto", devices=1)

current.obsm["X_miracle"] = miracle.get_latent_representation(current)
model_dir = miracle.save()
```

Current and replay batch names must be disjoint. If batch labels overlap across
steps, prefix them before calling `setup_continual`.
