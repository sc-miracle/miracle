# Installation

MIRACLE requires Python 3.10 or later. The project is tested primarily with
Python 3.11, PyTorch 2.5, Lightning 2.4+, Scanpy, AnnData, MuData, and
`scmidas==0.3.0`.

## Create an Environment

```bash
conda create -n scmiracle python=3.11
conda activate scmiracle
pip install scmiracle==0.1.0
```


This installs the `scmiracle` package and its runtime dependencies declared in
`pyproject.toml`, including MIDAS, PyTorch, Lightning, Scanpy, AnnData, and
MuData.

## Verify the API

Run a minimal import check after installation:

```bash
python - <<'PY'
from scmiracle.model import MIRACLE
from scmiracle.download_data import list_available_datasets

print(MIRACLE)
list_available_datasets()
PY
```

The supported public entry point is currently `from scmiracle.model import
MIRACLE`. The package-level shortcut `scmiracle.MIRACLE` is not exported by the
current `__init__.py`.

## GPU Notes

MIRACLE delegates training to the MIDAS/Lightning stack. Use
`accelerator="auto"` for portable CPU/GPU selection, or pass explicit Lightning
arguments through `MIRACLE.train(...)` when running on a managed GPU cluster.
