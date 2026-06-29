from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Literal, Optional, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.io import mmread
from scipy.sparse import issparse
from sklearn.neighbors import BallTree
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

try:
    import scmidas
    import scmidas.data as _scmidas_data
    from scmidas.config import load_config
    from scmidas.model import MIDAS as _MIDAS
    from scmidas.nn import transform_registry
    from scmidas.utils import detach_tensors, filter_keys
    _SCMIDAS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - used when building docs without runtime deps
    scmidas = None
    _scmidas_data = None
    _SCMIDAS_IMPORT_ERROR = exc

    def load_config():
        _require_scmidas()

    def detach_tensors(*args, **kwargs):
        _require_scmidas()

    def filter_keys(*args, **kwargs):
        _require_scmidas()

    class _MissingTransformRegistry:
        def get(self, *args, **kwargs):
            _require_scmidas()

    class _MissingMIDAS:
        @staticmethod
        def setup_mudata(*args, **kwargs):
            _require_scmidas()

        def __init__(self, *args, **kwargs):
            _require_scmidas()

    transform_registry = _MissingTransformRegistry()
    _MIDAS = _MissingMIDAS

if _scmidas_data is None:
    class _BasicModDatasetBase:
        pass
else:
    _BasicModDatasetBase = _scmidas_data.BasicModDataset


def _require_scmidas() -> None:
    """Raise a clear runtime error when MIDAS is unavailable."""
    if _SCMIDAS_IMPORT_ERROR is not None:
        raise ImportError(
            "MIRACLE requires scmidas with the MIDAS runtime modules installed. "
            "Install the package dependencies from pyproject.toml before using "
            "training or inference APIs."
        ) from _SCMIDAS_IMPORT_ERROR


class ReplayCurrentAlternatingSampler(Sampler):
    """Alternate replay and current-task batches from a concatenated dataset.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self, dataset: ConcatDataset, replay_dataset_count: int, batch_size: int):
        """Initialize the sampler for continual training.

        Args:
            dataset: Concatenated dataset containing replay subsets first and
                current-task subsets after them.
            replay_dataset_count: Number of child datasets at the beginning of
                ``dataset`` that belong to replay memory.
            batch_size: Number of examples to draw from one child dataset at a
                time before switching to the next source.

        Returns:
            None.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.replay_dataset_count = replay_dataset_count
        self.current_dataset_count = self.number_of_datasets - replay_dataset_count
        if self.current_dataset_count <= 0:
            raise ValueError("At least one current-task dataset is required.")
        if not 0 <= self.replay_dataset_count <= self.number_of_datasets:
            raise ValueError("replay_dataset_count must be between 0 and the total dataset count.")
        self.largest_current_dataset_size = max(
            [dataset.datasets[i].size for i in range(self.replay_dataset_count, self.number_of_datasets)]
        )

    def __len__(self):
        """Return the number of sampled indices produced in one epoch.

        Args:
            None.

        Returns:
            The total number of sample indices generated per epoch.
        """
        rounds = int(np.ceil(self.largest_current_dataset_size / self.batch_size))
        batches_per_round = int(self.current_dataset_count > 0) + int(self.replay_dataset_count > 0)
        return self.batch_size * rounds * batches_per_round

    def __iter__(self):
        """Yield indices that alternate between current and replay subsets.

        Args:
            None.

        Returns:
            An iterator of dataset indices for one sampling epoch.
        """
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = torch.utils.data.RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            sampler_iterators.append(iter(sampler))

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        final_samples_list = []
        replay_indices = np.arange(self.replay_dataset_count)
        current_indices = np.arange(self.replay_dataset_count, self.number_of_datasets)

        rounds = int(np.ceil(self.largest_current_dataset_size / self.batch_size))
        for _ in range(rounds):
            current_idx = int(np.random.choice(current_indices))
            final_samples_list.extend(
                self._get_sample_list(
                    push_index_val[current_idx], samplers_list[current_idx], sampler_iterators, current_idx
                )
            )

            if self.replay_dataset_count > 0:
                replay_idx = int(np.random.choice(replay_indices))
                final_samples_list.extend(
                    self._get_sample_list(
                        push_index_val[replay_idx], samplers_list[replay_idx], sampler_iterators, replay_idx
                    )
                )
        return iter(final_samples_list)

    def _get_sample_list(self, push_index_val, sampler, sampler_iterators, dataset_idx):
        """Draw one batch worth of indices from a single child dataset.

        Args:
            push_index_val: Global index offset for the child dataset inside the
                concatenated dataset.
            sampler: Random sampler associated with the child dataset.
            sampler_iterators: Mutable list of active sampler iterators.
            dataset_idx: Index of the child dataset to sample from.

        Returns:
            A list of global sample indices for one batch.
        """
        cur_batch_sampler = sampler_iterators[dataset_idx]
        cur_samples = []
        for _ in range(self.batch_size):
            try:
                cur_sample_org = next(cur_batch_sampler)
            except StopIteration:
                sampler_iterators[dataset_idx] = iter(sampler)
                cur_batch_sampler = sampler_iterators[dataset_idx]
                cur_sample_org = next(cur_batch_sampler)
            cur_samples.append(cur_sample_org + push_index_val)
        return cur_samples


class _LazyAnnDataDataset(_BasicModDatasetBase):
    """AnnData-backed dataset that densifies rows on demand."""

    def __init__(self, adata: ad.AnnData, use_layer: str = "X"):
        super().__init__()
        self.data = adata.X if use_layer == "X" else adata.layers[use_layer]
        self.size = int(adata.n_obs)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> np.ndarray:
        row = self.data[idx]
        if issparse(row):
            row = row.toarray()
        return np.asarray(row, dtype=np.float32).reshape(-1)


class _LazyAlignedAnnDataDataset(_BasicModDatasetBase):
    """Read selected AnnData rows and align features only when a row is requested."""

    def __init__(self, adata: ad.AnnData, row_indices: np.ndarray, target_vars: Sequence[str]):
        super().__init__()
        self.data = adata.X
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        source_vars = adata.var_names.astype(str)
        target_index = pd.Index(target_vars, dtype=str)
        self.source_to_target = target_index.get_indexer(source_vars)
        if np.any(self.source_to_target < 0):
            missing = source_vars[self.source_to_target < 0].tolist()
            raise ValueError(f"Source features are absent from the target layout: {missing[:3]}")
        self.target_size = len(target_vars)
        self.identity = (
            len(source_vars) == self.target_size
            and np.array_equal(self.source_to_target, np.arange(self.target_size))
        )
        self.observed_mask = np.zeros(self.target_size, dtype=np.float32)
        self.observed_mask[self.source_to_target] = 1.0
        self.size = len(self.row_indices)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> np.ndarray:
        row = self.data[int(self.row_indices[idx])]
        if issparse(row):
            row = row.toarray()
        source = np.asarray(row, dtype=np.float32).reshape(-1)
        if self.identity:
            return source
        aligned = np.zeros(self.target_size, dtype=np.float32)
        aligned[self.source_to_target] = source
        return aligned


class _LazyMultiModalDataset(Dataset):
    """MIDAS-compatible multimodal dataset backed by lazy aligned row readers."""

    def __init__(self, data, mod_ids, masks=None, transform=None):
        self.data = data
        self.mod_id_dict = mod_ids
        self.mask = masks or None
        self.transform = transform or {}
        self.size = len(next(iter(data.values())))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        items = {'x': {}, 's': {}, 'e': {}}
        for modality, dataset in self.data.items():
            value = dataset[idx]
            if modality in self.transform:
                value = transform_registry.get(self.transform[modality])(value)
            items['x'][modality] = value
            items['s'][modality] = np.array([self.mod_id_dict[modality]], dtype=np.int64)
        items['s']['joint'] = np.array([self.mod_id_dict['joint']], dtype=np.int64)
        if self.mask:
            items['e'].update(self.mask)
        return items


def _enable_lazy_scmidas_anndata() -> None:
    """Override scmidas AnnData loading to avoid eager full-matrix densification."""
    _require_scmidas()
    if _scmidas_data.modDataset_map.get("anndata") is _LazyAnnDataDataset:
        return
    _scmidas_data.modDataset_map["anndata"] = _LazyAnnDataDataset


class _MIDASContinual(_MIDAS):
    """MIDAS variant with replay-aware weighting and alternating sampling.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(
        self,
        *args,
        n_cells_orig: Optional[Sequence[int]] = None,
        batch_num_rep: int = 0,
        prepared_data: Optional[Dict] = None,
        **kwargs,
    ):
        """Store original cell counts and replay batch metadata.

        Args:
            *args: Positional arguments forwarded to ``scmidas.model.MIDAS``.
            n_cells_orig: Original cell counts for each batch before replay
                subsampling, aligned to the merged training batches.
            batch_num_rep: Number of replay batches in the merged continual
                dataset.
            **kwargs: Keyword arguments forwarded to ``scmidas.model.MIDAS``.

        Returns:
            None.
        """
        if prepared_data is None:
            super().__init__(*args, **kwargs)
        else:
            for key, value in prepared_data.items():
                setattr(self, key, value)
            # MIDAS(mdata=None) initializes the network from the prepared
            # instance attributes without scanning or slicing a MuData object.
            super().__init__(None)
        self.n_cells_orig = list(n_cells_orig or [])
        self.batch_num_rep = int(batch_num_rep)

    def _calc_rnt(self, batch, batch_idx: int) -> float:
        """Compute the replay-normalized training weight for the current batch.

        Args:
            batch: Current minibatch in MIDAS internal format.
            batch_idx: Zero-based minibatch index within the epoch.

        Returns:
            A scalar weight used to rebalance replay and current batches.
        """
        if self.batch_num_rep == 0:
            return 1.0
        subset_id = int(batch['s']['joint'][0].item())
        total_cells = float(sum(self.n_cells_orig))
        base = float(self.n_cells_orig[subset_id]) / total_cells
        if self.batch_num_rep == 0 or batch_idx % 2 < 1:
            return base
        return base * self.batch_num_rep

    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Run one continual-learning training step with weighted replay updates.

        Args:
            batch: Current minibatch in MIDAS internal format.
            batch_idx: Zero-based minibatch index within the epoch.

        Returns:
            The weighted network loss tensor for the step.
        """
        x_r_pre, s_r_pre, z_mu, z_logvar, z, c, u, z_uni, c_all = self.net(batch)
        c_all['joint'] = c

        recon_loss, recon_dict = self.calc_recon_loss(
            batch['x'], batch['s']['joint'], batch['e'],
            x_r_pre, s_r_pre,
            filter_keys(self.__dict__, 'distribution_dec_'),
            filter_keys(self.__dict__, 'lam_recon_')
        )
        recon_loss *= self.lam_recon

        kld_loss = self.calc_kld_z_loss(
            self.dim_c, self.dim_u, self.lam_kld_c, self.lam_kld_u, z_mu, z_logvar
        ) * self.lam_kld
        consistency_loss = self.calc_consistency_loss(z_uni) * self.lam_alignment
        loss_net = recon_loss + kld_loss + consistency_loss

        rnt = self._calc_rnt(batch, batch_idx)
        for _ in range(self.n_iter_disc):
            self._train_discriminator_weighted(c_all, batch['s'], rnt)

        s_pred = self.dsc(c_all)
        loss_dsc = self.calc_dsc_loss(s_pred, batch['s']) * self.lam_dsc
        loss_net = loss_net - loss_dsc * self.lam_adv
        loss_net = loss_net * rnt

        self.update_model(loss_net, self.net, self.net_optim, self.grad_clip)
        self.log_losses(recon_loss, kld_loss, consistency_loss, loss_net, loss_dsc, recon_dict)
        return loss_net

    def _train_discriminator_weighted(self, c_all, targets, rnt: float):
        """Update the discriminator using the same replay weight as the generator step.

        Args:
            c_all: Latent representations passed to the discriminator.
            targets: Batch/domain targets for adversarial training.
            rnt: Replay-normalized training weight for the current minibatch.

        Returns:
            None.
        """
        s_pred = self.dsc(detach_tensors(c_all))
        loss_dsc = self.calc_dsc_loss(s_pred, targets) * self.lam_dsc
        loss_dsc = loss_dsc * rnt
        self.update_model(loss_dsc, self.dsc, self.dsc_optim, self.grad_clip)

    def train_dataloader(self):
        """Build a dataloader that alternates replay and current-task subsets.

        Args:
            None.

        Returns:
            A PyTorch dataloader configured for continual training.
        """
        dataset = ConcatDataset(self.datalist)
        sampler = ReplayCurrentAlternatingSampler(
            dataset,
            replay_dataset_count=self.batch_num_rep,
            batch_size=self.batch_size,
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class MIRACLE:
    """High-level MuData wrapper around MIDAS for continual multimodal learning.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(
        self,
        *,
        batch_key: str = "batch",
        configs: Optional[dict] = None,
        batch_size: int = 128,
        save_model_path: str = "./saved_models/miracle",
    ):
        """Initialize the MIRACLE training helper.

        Args:
            batch_key: Observation column used as the batch identifier.
            configs: Optional MIDAS configuration dictionary. When omitted, the
                default SCMIDAS config is loaded.
            batch_size: Minibatch size used for training and latent inference.
            save_model_path: Default directory used when saving checkpoints.

        Returns:
            None.
        """
        _enable_lazy_scmidas_anndata()
        self.batch_key = batch_key
        self.configs = copy.deepcopy(configs) if configs is not None else load_config()
        self.batch_size = batch_size
        self.save_model_path = save_model_path

        self.model = None
        self.mode = None
        self.current_mdata = None
        self.replay_mdata = None
        self.prev_model_dir = None
        self.n_cells_orig: List[int] = []
        self.last_replay_source = None
        self.replay_batch_cell_counts_orig: Dict[str, int] = {}
        self._lazy_continual = False
        self._lazy_training_sources = None

    @staticmethod
    def _to_jsonable(value):
        """Convert NumPy-heavy metadata into JSON-serializable Python objects.

        Args:
            value: Arbitrary nested object that may contain NumPy scalars,
                arrays, lists, tuples, or dictionaries.

        Returns:
            A JSON-serializable Python object.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): MIRACLE._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MIRACLE._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _count_cells(mdata: mu.MuData) -> int:
        """Return the number of observations from top-level MuData metadata.

        Args:
            mdata: Input MuData object.

        Returns:
            The total number of observations.
        """
        return int(mdata.n_obs)

    @staticmethod
    def read_named_csv_column(path: Path | str, column: str = "x") -> List[str]:
        """Read a named column from a CSV file, with a fallback to the last column.

        Args:
            path: CSV file path.
            column: Preferred column name to read.

        Returns:
            A list of string values from the requested column.
        """
        path = Path(path)
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            df = pd.read_csv(path)
        values = df[column] if column in df.columns else df.iloc[:, -1]
        return values.astype(str).tolist()

    @staticmethod
    def attach_feature_and_label_metadata(
        mdata: mu.MuData,
        *,
        feature_paths_by_mod: Optional[Dict[str, Sequence[Path | str]]] = None,
        labels: Optional[Sequence[str]] = None,
        label_key: str = "label",
    ) -> mu.MuData:
        """Attach feature names and optional labels to a MuData object.

        Args:
            mdata: Input MuData object to annotate.
            feature_paths_by_mod: Optional mapping from modality name to one or
                more CSV files containing feature names.
            labels: Optional cell labels aligned to observations.
            label_key: Observation column name used for labels.

        Returns:
            The same MuData object with updated feature and label metadata.
        """
        feature_name_map = {}
        if feature_paths_by_mod:
            for mod, adata in mdata.mod.items():
                for feat_path in feature_paths_by_mod.get(mod, []):
                    feat_path = Path(feat_path)
                    if not feat_path.exists():
                        continue
                    feature_names = MIRACLE.read_named_csv_column(feat_path)
                    if len(feature_names) != adata.n_vars:
                        continue
                    adata.var_names = pd.Index(feature_names, dtype=str)
                    adata.var_names_make_unique()
                    feature_name_map[mod] = feature_names
                    break

        if labels is not None:
            labels = list(map(str, labels))
            for adata in mdata.mod.values():
                if len(labels) == adata.n_obs:
                    adata.obs[label_key] = pd.Categorical(labels)

        mdata.update()
        if labels is not None and len(labels) == mdata.n_obs:
            mdata.obs[label_key] = pd.Categorical(labels)
        if feature_name_map:
            mdata.uns["feature_names"] = feature_name_map
        return mdata

    @staticmethod
    def prefix_batch_and_obs_names(
        mdata: mu.MuData,
        *,
        batch_key: str = "batch",
        batch_prefix: Optional[str] = None,
        obs_prefix: Optional[str] = None,
        label_key: str = "label",
    ) -> mu.MuData:
        """Prefix batch labels and observation names while preserving top-level metadata.

        Args:
            mdata: Input MuData object.
            batch_key: Observation column containing batch labels.
            batch_prefix: Optional prefix added to each batch label.
            obs_prefix: Optional prefix added to each observation name.
            label_key: Observation column containing labels.

        Returns:
            A copied MuData object with prefixed identifiers.
        """
        mod_map = {}
        for mod, adata in mdata.mod.items():
            adata = adata.copy()
            if batch_prefix is not None and batch_key in adata.obs.columns:
                adata.obs[batch_key] = adata.obs[batch_key].astype(str).map(
                    lambda x: f"{batch_prefix}_{x}"
                )
            if obs_prefix is not None:
                adata.obs_names = [f"{obs_prefix}_{cid}" for cid in adata.obs_names.astype(str)]
                adata.obs_names_make_unique()
            mod_map[mod] = adata

        out = mu.MuData(mod_map)
        for key, value in mdata.uns.items():
            out.uns[key] = copy.deepcopy(value)
        out.update()
        if batch_prefix is None and obs_prefix is None:
            return out

        batch_top = pd.Series(index=out.obs_names, dtype=object)
        label_top = pd.Series(index=out.obs_names, dtype=object)
        for adata in out.mod.values():
            if batch_key in adata.obs.columns:
                batch_top.loc[adata.obs_names] = adata.obs[batch_key].astype(str).values
            if label_key in adata.obs.columns:
                label_top.loc[adata.obs_names] = adata.obs[label_key].astype(str).values
        if not batch_top.isna().all():
            out.obs[batch_key] = pd.Categorical(batch_top.reindex(out.obs_names).astype(str).values)
        if not label_top.isna().all():
            out.obs[label_key] = pd.Categorical(label_top.reindex(out.obs_names).astype(str).values)
        return out

    @classmethod
    def load_mtx_dir_as_mudata(
        cls,
        data_dir: Path | str,
        *,
        label_path: Optional[Path | str] = None,
        batch_prefix: Optional[str] = None,
        batch_key: str = "batch",
        label_key: str = "label",
    ) -> mu.MuData:
        """Load an original MIRACLE mtx-format directory and convert it into MuData.

        Args:
            data_dir: Dataset directory containing ``feat/`` and ``subset_*``
                folders in the original MIRACLE input layout.
            label_path: Optional CSV file containing cell labels indexed by cell
                name. When provided, labels are attached to each modality and to
                the top-level ``mdata.obs``.
            batch_prefix: Optional prefix added to each subset batch name after
                loading, for example ``ref`` or ``query``.
            batch_key: Observation column used for batch labels.
            label_key: Observation column used for cell labels.

        Returns:
            A MuData object ready to be used with ``MIRACLE.setup_mudata`` or
            ``MIRACLE.setup_continual``.
        """
        data_dir = Path(data_dir)
        feat_dir = data_dir / "feat"
        if not feat_dir.exists():
            raise FileNotFoundError(f"Missing feat directory: {feat_dir}")

        feat_dims_path = feat_dir / "feat_dims.toml"
        if not feat_dims_path.exists():
            raise FileNotFoundError(f"Missing feat_dims.toml: {feat_dims_path}")
        feat_dims = tomllib.loads(feat_dims_path.read_text())

        labels = None
        if label_path is not None:
            labels = pd.read_csv(label_path, index_col=0).iloc[:, 0].astype(str)

        parts = []
        subset_dirs = sorted(data_dir.glob("subset_*"), key=lambda p: int(p.name.split("_")[-1]))
        for subset_dir in subset_dirs:
            cell_path = subset_dir / "cell_names.csv"
            mat_dir = subset_dir / "mat"
            if not cell_path.exists():
                raise FileNotFoundError(f"Missing cell_names.csv: {cell_path}")
            if not mat_dir.exists():
                raise FileNotFoundError(f"Missing mat directory: {mat_dir}")

            cell_names = pd.read_csv(cell_path, index_col=0).iloc[:, 0].astype(str)
            mod_map = {}
            for mod_file in sorted(mat_dir.glob("*.mtx")):
                mod = mod_file.stem
                feat_name_path = feat_dir / f"feat_names_{mod}.csv"
                if not feat_name_path.exists():
                    raise FileNotFoundError(f"Missing feature names for modality {mod}: {feat_name_path}")

                X = mmread(mod_file).tocsr()
                feat_names = pd.read_csv(feat_name_path, index_col=0).iloc[:, 0].astype(str)
                adata = ad.AnnData(X)
                adata.obs_names = pd.Index(cell_names.values, dtype=str)
                adata.var_names = pd.Index(feat_names.iloc[:X.shape[1]].values, dtype=str)
                adata.obs[batch_key] = pd.Categorical([subset_dir.name] * adata.n_obs)
                if labels is not None:
                    adata.obs[label_key] = pd.Categorical(
                        labels.reindex(adata.obs_names).fillna("Unknown").astype(str).values
                    )
                mod_map[mod] = adata

            part = mu.MuData(mod_map)
            part.uns["feat_dims"] = copy.deepcopy(feat_dims)
            part.update()
            part = cls.prefix_batch_and_obs_names(
                part,
                batch_key=batch_key,
                batch_prefix=batch_prefix,
                label_key=label_key,
            )
            parts.append(part)

        mdata = cls._concat_mudata(parts)
        mdata.obs[batch_key] = cls.get_obs_series(mdata, batch_key).reindex(mdata.obs_names).astype(str).values
        if labels is not None:
            mdata.obs[label_key] = cls.get_obs_series(mdata, label_key).reindex(mdata.obs_names).astype(str).values
        return mdata

    def _count_cells_per_batch(self, mdata: mu.MuData) -> List[int]:
        """Count observations for each batch in the order expected by the model.

        Args:
            mdata: Input MuData object.

        Returns:
            A list of per-batch cell counts aligned to MIRACLE batch order.
        """
        batch_names = mdata.uns.get("_scmidas", {}).get("batch_names")
        if batch_names is None or len(batch_names) == 0:
            discovered = set()
            for adata in mdata.mod.values():
                discovered.update(adata.obs[self.batch_key].astype(str).unique().tolist())
            batch_names = sorted(discovered)

        counts = []
        for batch_name in batch_names:
            count = None
            for adata in mdata.mod.values():
                mask = adata.obs[self.batch_key].astype(str) == batch_name
                n_obs = int(mask.sum())
                if n_obs > 0:
                    count = n_obs
                    break
            if count is None:
                raise ValueError(f"Unable to count cells for batch {batch_name!r}.")
            counts.append(count)
        return counts

    def _count_batches(self, mdata: mu.MuData) -> int:
        """Return the number of unique batches from top-level MuData metadata.

        Args:
            mdata: Input MuData object.

        Returns:
            The number of distinct batch labels.
        """
        return int(mdata.obs[self.batch_key].astype(str).nunique())

    def _get_batch_names(self, mdata: mu.MuData) -> List[str]:
        """Get batch names from cached MIDAS metadata or infer them from modalities.

        Args:
            mdata: Input MuData object.

        Returns:
            A sorted list of batch names.
        """
        batch_names = mdata.uns.get("_scmidas", {}).get("batch_names")
        if batch_names is not None and len(batch_names) > 0:
            return list(batch_names)
        discovered = set()
        for adata in mdata.mod.values():
            discovered.update(adata.obs[self.batch_key].astype(str).unique().tolist())
        return sorted(discovered)

    @staticmethod
    def _resolve_checkpoint_path(path: Path | str) -> tuple[Path, Path]:
        """Resolve a checkpoint input as either a model directory or a model.pt file."""
        raw = Path(path)
        if raw.is_file():
            return raw, raw.parent
        candidate = raw / "model.pt"
        if candidate.exists():
            return candidate, raw
        raise FileNotFoundError(
            f"Checkpoint not found. Expected either a file path or a directory containing model.pt: {raw}"
        )

    def _get_obs_batch_series(self, mdata: mu.MuData):
        """Return the batch annotation as a top-level observation series.

        Args:
            mdata: Input MuData object.

        Returns:
            A pandas Series containing batch labels.
        """
        return self.get_obs_series(mdata, self.batch_key)

    @staticmethod
    def get_obs_series(mdata: mu.MuData, key: str) -> pd.Series:
        """Resolve an observation annotation from top-level or modality-specific columns.

        Args:
            mdata: Input MuData object.
            key: Observation key such as ``batch`` or ``label``.

        Returns:
            A top-level observation series aligned to ``mdata.obs_names``.
        """
        if key in mdata.obs.columns:
            series = mdata.obs[key]
            if not series.isna().all():
                values = series.astype(str)
                if any(not v.endswith("_nan") for v in values):
                    return values

        batch_cols = [col for col in mdata.obs.columns if col.endswith(f":{key}")]
        if batch_cols:
            frame = mdata.obs[batch_cols]
            series = frame.bfill(axis=1).iloc[:, 0]
            if not series.isna().all():
                return series.astype(str)

        merged = pd.Series(index=mdata.obs_names, dtype=object)
        found = False
        for adata in mdata.mod.values():
            if key not in adata.obs.columns:
                continue
            values = adata.obs[key].astype(str)
            merged.loc[adata.obs_names] = merged.loc[adata.obs_names].where(
                merged.loc[adata.obs_names].notna(),
                values,
            )
            found = True
        if found and not merged.isna().all():
            return merged.astype(str)
        raise KeyError(f"Observation key {key!r} not found in mdata.obs or any modality-specific obs.")

    def _get_original_cell_counts(self, mdata: mu.MuData) -> Dict[str, int]:
        """Read original per-batch cell counts from replay metadata when available.

        Args:
            mdata: Input MuData object.

        Returns:
            A mapping from batch name to original cell count.
        """
        meta = mdata.uns.get("_miracle_replay_meta", {})
        stored = meta.get("batch_cell_counts_orig")
        if isinstance(stored, dict):
            return {str(k): int(v) for k, v in stored.items()}
        if stored is not None:
            batch_names = self._get_batch_names(mdata)
            if len(batch_names) != 1:
                raise ValueError(
                    "Scalar batch_cell_counts_orig metadata is only supported for single-batch MuData."
                )
            return {str(batch_names[0]): int(stored)}
        return dict(zip(self._get_batch_names(mdata), self._count_cells_per_batch(mdata)))

    def _get_training_cell_counts_orig(self, mdata: mu.MuData, replay_mdata: Optional[mu.MuData] = None) -> List[int]:
        """Build original per-batch cell counts aligned to the merged training batches.

        Args:
            mdata: Merged MuData used for training.
            replay_mdata: Optional replay MuData used to recover original counts
                for replay batches.

        Returns:
            A list of original cell counts aligned to merged batch order.
        """
        batch_names = self._get_batch_names(mdata)
        replay_counts = self._get_original_cell_counts(replay_mdata) if replay_mdata is not None else {}
        current_counts = self._get_original_cell_counts(self.current_mdata if self.current_mdata is not None else mdata)
        counts = []
        for batch_name in batch_names:
            if batch_name in replay_counts:
                counts.append(replay_counts[batch_name])
            elif batch_name in current_counts:
                counts.append(current_counts[batch_name])
            else:
                raise ValueError(f"Missing original cell count for batch {batch_name!r}.")
        return counts

    def _get_training_cell_count_map_orig(
        self,
        mdata: mu.MuData,
        replay_mdata: Optional[mu.MuData] = None,
    ) -> Dict[str, int]:
        """Return original per-batch cell counts as a mapping for metadata export.

        Args:
            mdata: Merged MuData used for training.
            replay_mdata: Optional replay MuData used to recover original counts
                for replay batches.

        Returns:
            A mapping from batch name to original cell count.
        """
        batch_names = self._get_batch_names(mdata)
        replay_counts = self._get_original_cell_counts(replay_mdata) if replay_mdata is not None else {}
        current_source = self.current_mdata if self.current_mdata is not None else mdata
        current_counts = self._get_original_cell_counts(current_source)
        counts = {}
        for batch_name in batch_names:
            if batch_name in replay_counts:
                counts[batch_name] = replay_counts[batch_name]
            elif batch_name in current_counts:
                counts[batch_name] = current_counts[batch_name]
            else:
                raise ValueError(f"Missing original cell count for batch {batch_name!r}.")
        return counts

    def _validate_disjoint_batch_names(self, current_mdata: mu.MuData, replay_mdata: mu.MuData) -> None:
        """Validate that current and replay MuData objects use disjoint batch names.

        Args:
            current_mdata: MuData for the current continual-learning step.
            replay_mdata: Replay MuData sampled from previous steps.

        Returns:
            None.
        """
        current_batches = set(self._get_batch_names(current_mdata))
        replay_batches = set(self._get_batch_names(replay_mdata))
        overlap = sorted(current_batches & replay_batches)
        if overlap:
            raise ValueError(
                "Current and replay MuData contain overlapping batch names: "
                f"{overlap}. Prefix batch names (for example with "
                "MIRACLE.prefix_batch_and_obs_names(...)) before continual training."
            )

    def _should_inherit_dsc(
        self,
        ckpt: Dict,
        ckpt_dir: Path,
        replay_mdata: mu.MuData,
        current_mdata: mu.MuData,
    ) -> tuple[bool, str]:
        """Decide whether discriminator weights can be transferred safely."""
        source_batch_names = self._infer_checkpoint_batch_names(ckpt, ckpt_dir)
        if not source_batch_names:
            return False, "missing checkpoint batch_names metadata"

        replay_batch_names = self._get_batch_names(replay_mdata)
        if source_batch_names != replay_batch_names:
            return False, "source checkpoint batch_names do not match replay batch_names"

        target_batch_names = replay_batch_names + self._get_batch_names(current_mdata)
        if target_batch_names[:len(source_batch_names)] != source_batch_names:
            return False, "target batch order does not preserve replay batch prefix"

        return True, "ok"

    @classmethod
    def _infer_checkpoint_batch_names(cls, ckpt: Dict, ckpt_dir: Path) -> List[str]:
        """Infer checkpoint batch order from embedded metadata or setup.json."""
        metadata = ckpt.get("metadata")
        if isinstance(metadata, dict):
            batch_names = metadata.get("batch_names")
            if batch_names:
                # print("[dsc/infer] source batch_names from checkpoint metadata")
                return list(map(str, batch_names))

        setup_path = ckpt_dir / "setup.json"
        if setup_path.exists():
            try:
                with open(setup_path, "rb") as f:
                    saved_setup = json.load(f)
                batch_names = list(map(str, saved_setup.get("batch_names", [])))
                if batch_names:
                    # print(f"[dsc/infer] source batch_names from {setup_path}")
                    return batch_names
            except Exception as exc:
                # print(f"[dsc/infer] failed to read {setup_path}: {exc}")
                pass

        return []

    @staticmethod
    def _feature_segment_key(name: str) -> str:
        return name.split("-", 1)[0] if "-" in name else name

    @classmethod
    def _build_target_feature_layout(
        cls,
        adatas: Sequence[ad.AnnData],
        ref_dims: Optional[Sequence[int]],
    ) -> tuple[List[str], List[int]]:
        """Keep reference features first and append new features per segment."""
        ref_vars = adatas[0].var_names.astype(str).tolist()
        if ref_dims is not None and len(ref_dims) > 1:
            if int(sum(ref_dims)) != len(ref_vars):
                raise ValueError(
                    f"Reference feat_dims sum {sum(ref_dims)} does not match n_vars={len(ref_vars)}."
                )
            segment_keys = []
            segment_ref_vars = []
            start = 0
            for dim in ref_dims:
                end = start + int(dim)
                segment = ref_vars[start:end]
                if not segment:
                    raise ValueError("Encountered empty segmented feature block.")
                segment_keys.append(cls._feature_segment_key(segment[0]))
                segment_ref_vars.append(segment)
                start = end
            if len(set(segment_keys)) != len(segment_keys):
                raise ValueError(
                    "Segmented feat_dims requires distinct feature prefixes before '-'; "
                    f"got {segment_keys}."
                )

            extras = {key: [] for key in segment_keys}
            seen = set(ref_vars)
            for adata_in in adatas[1:]:
                for name in adata_in.var_names.astype(str):
                    if name in seen:
                        continue
                    key = cls._feature_segment_key(name)
                    if key not in extras:
                        raise ValueError(
                            f"New feature {name!r} maps to unknown segment {key!r}."
                        )
                    extras[key].append(name)
                    seen.add(name)

            target_vars = []
            target_dims = []
            for key, segment in zip(segment_keys, segment_ref_vars):
                combined = segment + extras[key]
                target_vars.extend(combined)
                target_dims.append(len(combined))
            return target_vars, target_dims

        seen = set(ref_vars)
        new_vars = []
        for adata_in in adatas[1:]:
            for name in adata_in.var_names.astype(str):
                if name not in seen:
                    new_vars.append(name)
                    seen.add(name)
        target_vars = ref_vars + new_vars
        return target_vars, [len(target_vars)]

    @staticmethod
    def _align_mask_to_target(mask_value, source_vars: Sequence[str], target_vars: Sequence[str]) -> np.ndarray:
        mask_arr = np.asarray(mask_value, dtype=np.float32).reshape(-1)
        if mask_arr.shape[0] != len(source_vars):
            raise ValueError(
                f"Mask length {mask_arr.shape[0]} does not match source vars {len(source_vars)}."
            )
        positions = pd.Index(source_vars, dtype=str).get_indexer(pd.Index(target_vars, dtype=str))
        aligned = np.zeros(len(target_vars), dtype=np.float32)
        present = positions >= 0
        aligned[present] = mask_arr[positions[present]]
        return aligned

    def _prepare_lazy_continual_data(
        self,
        replay_mdata: mu.MuData,
        current_mdata: mu.MuData,
        configs: Dict,
    ) -> Dict:
        """Build MIDAS datasets without concatenating source expression matrices."""
        prepared_configs = dict(configs)
        if replay_mdata.isbacked or current_mdata.isbacked:
            # h5py-backed handles should not be inherited by multiple forked
            # workers. A worker-safe reopen strategy can be added separately.
            prepared_configs['num_workers'] = 0
            prepared_configs['persistent_workers'] = False
        replay_batches = self._get_batch_names(replay_mdata)
        current_batches = self._get_batch_names(current_mdata)
        batch_specs = [("replay", replay_mdata, b) for b in replay_batches]
        batch_specs.extend(("current", current_mdata, b) for b in current_batches)

        mods = sorted(set(replay_mdata.mod) | set(current_mdata.mod))
        target_vars = {}
        dims_x = {}
        for mod in mods:
            adatas = []
            if mod in replay_mdata.mod:
                adatas.append(replay_mdata.mod[mod])
            if mod in current_mdata.mod:
                adatas.append(current_mdata.mod[mod])
            dims_source = replay_mdata if mod in replay_mdata.mod else current_mdata
            ref_dims = dims_source.uns.get("feat_dims", {}).get(mod)
            target_vars[mod], dims_x[mod] = self._build_target_feature_layout(adatas, ref_dims)
            ref_n_vars = int(adatas[0].n_vars)
            current_n_vars = int(adatas[-1].n_vars)
            added = int(len(target_vars[mod]) - ref_n_vars)
            # print(
            #     f"[lazy/feature] mod={mod} ref_n_vars={ref_n_vars} "
            #     f"current_n_vars={current_n_vars} target_n_vars={len(target_vars[mod])} "
            #     f"added_vs_ref={added} dims_x={dims_x[mod]}"
            # )

        effective_transform = {'atac': 'binarize'} if 'atac' in mods else {}
        modality_counts = {}
        datalist = []
        s_joint = []
        combs = []
        # print(
        #     f"[lazy/setup] replay_batches={replay_batches} current_batches={current_batches} "
        #     f"mods={mods} transform={effective_transform}"
        # )
        for joint_id, (_, source, batch_name) in enumerate(batch_specs):
            batch_data = {}
            batch_masks = {}
            batch_s = {'joint': joint_id}
            batch_combs = []
            for mod in mods:
                if mod not in source.mod:
                    continue
                adata_in = source.mod[mod]
                values = adata_in.obs[self.batch_key].astype(str).to_numpy()
                row_indices = np.flatnonzero(values == str(batch_name))
                if len(row_indices) == 0:
                    continue
                batch_data[mod] = _LazyAlignedAnnDataDataset(
                    adata_in, row_indices, target_vars[mod]
                )
                observed_mask = batch_data[mod].observed_mask
                mod_id = modality_counts.get(mod, 0)
                batch_s[mod] = mod_id
                modality_counts[mod] = mod_id + 1
                batch_combs.append(mod)
                mask_key = f"mask_{batch_name}"
                if mask_key in adata_in.uns:
                    batch_masks[mod] = self._align_mask_to_target(
                        adata_in.uns[mask_key], adata_in.var_names.astype(str), target_vars[mod]
                    ) * observed_mask
                    mask_state = "aligned_uns_mask"
                else:
                    batch_masks[mod] = observed_mask
                    mask_state = "observed_mask_only"
                # print(
                #     f"[lazy/batch] joint={joint_id} batch={batch_name} mod={mod} "
                #     f"rows={len(row_indices)} mod_id={batch_s[mod]} "
                #     f"source_n_vars={adata_in.n_vars} target_n_vars={len(target_vars[mod])} "
                #     f"observed_features={int(observed_mask.sum())} mask={mask_state}"
                # )
            if not batch_data:
                raise ValueError(f"Batch {batch_name!r} has no modality data.")
            datalist.append(
                _LazyMultiModalDataset(batch_data, batch_s, batch_masks, effective_transform)
            )
            s_joint.append(batch_s)
            combs.append(batch_combs)

        dims_s = {mod: count for mod, count in modality_counts.items()}
        dims_s['joint'] = len(batch_specs)
        # print(f"[lazy/result] dims_x={dims_x}")
        # print(f"[lazy/result] dims_s={dims_s}")
        # print(f"[lazy/result] s_joint_head={s_joint[:3]} total_batches={len(s_joint)}")
        metadata_mdata = self._build_lazy_metadata_mdata(
            replay_mdata,
            current_mdata,
            target_vars=target_vars,
            dims_x=dims_x,
            batch_names=replay_batches + current_batches,
        )
        return {
            'configs': prepared_configs,
            'dims_x': dims_x,
            'dims_s': dims_s,
            'batch_names': replay_batches + current_batches,
            'datalist': datalist,
            's_joint': s_joint,
            'combs': combs,
            'mods': mods,
            'save_model_path': self.save_model_path,
            'batch_size': self.batch_size,
            'n_save': 500,
            'sampler_type': 'auto',
            'viz_umap_tb': False,
            '_mdata': metadata_mdata,
        }

    def _build_lazy_metadata_mdata(
        self,
        replay_mdata: mu.MuData,
        current_mdata: mu.MuData,
        *,
        target_vars: Dict[str, Sequence[str]],
        dims_x: Dict[str, Sequence[int]],
        batch_names: Sequence[str],
    ) -> mu.MuData:
        """Create an obs/var-only MuData used for ordering and metadata lookup."""
        mod_map = {}
        for mod, variables in target_vars.items():
            obs_parts = []
            var = pd.DataFrame(index=pd.Index(variables, dtype=str))
            dtype = np.float32
            for source in (replay_mdata, current_mdata):
                if mod not in source.mod:
                    continue
                adata_in = source.mod[mod]
                obs_parts.append(adata_in.obs.copy())
                dtype = adata_in.X.dtype
                incoming = adata_in.var.reindex(var.index)
                for column in incoming.columns:
                    if column not in var.columns:
                        var[column] = incoming[column]
                    else:
                        var[column] = var[column].where(var[column].notna(), incoming[column])
            obs = pd.concat(obs_parts, axis=0, sort=False)
            X = sp.csr_matrix((len(obs), len(variables)), dtype=dtype)
            mod_map[mod] = ad.AnnData(X=X, obs=obs, var=var)

        out = mu.MuData(mod_map)
        out.uns['feat_dims'] = {m: list(map(int, dims)) for m, dims in dims_x.items()}
        out.uns['_scmidas'] = {
            'batch_key': self.batch_key,
            'dims_x': out.uns['feat_dims'],
            'batch_names': list(batch_names),
            'mods': list(mod_map),
            'version': 1,
        }
        out.uns['_miracle_lazy_continual'] = True

        top_obs = pd.concat(
            [replay_mdata.obs.copy(), current_mdata.obs.copy()], axis=0, sort=False
        )
        for column in top_obs.columns:
            out.obs[column] = top_obs[column].reindex(out.obs_names).to_numpy()
        return out

    def setup_mudata(self, mdata: mu.MuData, *, save_model_path: Optional[str] = None):
        """Prepare a step-1 MIDAS model from a single MuData object.

        Args:
            mdata: Current-task MuData used for the first training step.
            save_model_path: Optional checkpoint directory overriding the
                instance default.

        Returns:
            The current ``MIRACLE`` instance.
        """
        self.mode = "step1"
        self.current_mdata = mdata
        self.replay_mdata = None
        if save_model_path is not None:
            self.save_model_path = save_model_path
        dims_x = mdata.uns.get("feat_dims")
        _MIDAS.setup_mudata(mdata, batch_key=self.batch_key, dims_x=dims_x)
        self.model = _MIDAS(
            mdata,
            configs=self.configs,
            batch_size=self.batch_size,
            save_model_path=self.save_model_path,
        )
        self.n_cells_orig = self._get_training_cell_counts_orig(mdata)
        return self

    def setup_continual(
        self,
        current_mdata: mu.MuData,
        *,
        replay_mdata: mu.MuData,
        prev_model_dir: str,
        save_model_path: Optional[str] = None,
        inherit_dsc: bool = True,
        lazy: bool = True,
    ):
        """Prepare a continual model from replay data, current data, and a previous checkpoint.

        Args:
            current_mdata: MuData for the current continual-learning step.
            replay_mdata: Replay MuData sampled from previous steps.
            prev_model_dir: Directory containing the previous step checkpoint.
            save_model_path: Optional checkpoint directory overriding the
                instance default.
            inherit_dsc: Whether to initialize the discriminator from the
                previous checkpoint when compatible weights are available.
            lazy: Build batch datasets directly from replay/current matrices
                without materializing a merged expression matrix. Defaults to
                ``True``. Set to ``False`` for the legacy eager behavior.

        Returns:
            The current ``MIRACLE`` instance.
        """
        self.mode = "continual"
        self.current_mdata = current_mdata
        self.replay_mdata = replay_mdata
        ckpt_path, ckpt_dir = self._resolve_checkpoint_path(prev_model_dir)
        self.prev_model_dir = str(ckpt_dir)
        if save_model_path is not None:
            self.save_model_path = save_model_path

        self._validate_disjoint_batch_names(current_mdata, replay_mdata)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        configs = ckpt.get("configs", self.configs)
        source_batch_names = self._infer_checkpoint_batch_names(ckpt, ckpt_dir)
        if source_batch_names:
            replay_batches_existing = self._get_batch_names(replay_mdata)
            if set(source_batch_names) == set(replay_batches_existing):
                replay_mdata.uns.setdefault("_scmidas", {})
                replay_mdata.uns["_scmidas"]["batch_names"] = list(source_batch_names)
                # print(f"[setup_continual] aligned replay batch order to checkpoint metadata: {source_batch_names}")
            else:
                # print(
                #     "[setup_continual] checkpoint batch_names do not match replay batch set; "
                #     f"checkpoint={source_batch_names} replay={replay_batches_existing}"
                # )
                pass
        replay_batches = self._get_batch_names(replay_mdata)
        current_batches = self._get_batch_names(current_mdata)
        # print(
        #     f"[setup_continual] lazy={lazy} inherit_dsc={inherit_dsc} "
        #     f"ckpt={ckpt_path} replay_batches={replay_batches} current_batches={current_batches}"
        # )
        # print(
        #     f"[setup_continual] replay_mods={list(replay_mdata.mod.keys())} "
        #     f"current_mods={list(current_mdata.mod.keys())}"
        # )
        if lazy:
            prepared = self._prepare_lazy_continual_data(replay_mdata, current_mdata, configs)
            self.n_cells_orig = self._get_training_cell_counts_orig(
                prepared['_mdata'], replay_mdata
            )
            model = _MIDASContinual(
                prepared_data=prepared,
                n_cells_orig=self.n_cells_orig,
                batch_num_rep=len(self._get_batch_names(replay_mdata)),
            )
            self._lazy_training_sources = (replay_mdata, current_mdata)
            self._lazy_continual = True
        else:
            merged = self._concat_mudata([replay_mdata, current_mdata])
            dims_x = merged.uns.get("feat_dims")
            _MIDAS.setup_mudata(merged, batch_key=self.batch_key, dims_x=dims_x)
            self.n_cells_orig = self._get_training_cell_counts_orig(merged, replay_mdata)
            model = _MIDASContinual(
                merged,
                configs=configs,
                batch_size=self.batch_size,
                save_model_path=self.save_model_path,
                n_cells_orig=self.n_cells_orig,
                batch_num_rep=len(self._get_batch_names(replay_mdata)),
            )
            self._lazy_training_sources = None
            self._lazy_continual = False
        # print(f"[setup_continual] n_cells_orig={self.n_cells_orig}")
        # print(f"[setup_continual] target_dims_x={model.dims_x}")
        # print(f"[setup_continual] target_dims_s={model.dims_s}")
        # print(f"[setup_continual] model.batch_names={getattr(model, 'batch_names', None)}")
        self._transfer_compatible_state(model.net, ckpt["net"], module_name="net")
        if inherit_dsc and "dsc" in ckpt:
            can_inherit_dsc, dsc_reason = self._should_inherit_dsc(ckpt, ckpt_dir, replay_mdata, current_mdata)
            # print(f"[setup_continual] dsc_transfer_allowed={can_inherit_dsc} reason={dsc_reason}")
            if can_inherit_dsc:
                self._transfer_compatible_state(model.dsc, ckpt["dsc"], module_name="dsc")
            else:
                warnings.warn(
                    f"Skipping discriminator transfer: {dsc_reason}. "
                    "Use fresh discriminator weights for continual training."
                )
                # print("[dsc] skipped previous-step transfer; using fresh discriminator weights")
                pass
        else:
            # print("[dsc] skipped previous-step transfer; using fresh discriminator weights")
            pass
        self.model = model
        return self

    def train(self, **kwargs):
        """Train the currently configured MIDAS model.

        Args:
            **kwargs: Keyword arguments forwarded to ``MIDAS.train``.

        Returns:
            The current ``MIRACLE`` instance.
        """
        if self.model is None:
            raise RuntimeError("Call setup_mudata(...) or setup_continual(...) before train().")
        self.model.train(**kwargs)
        return self

    def save(self, dir_path: Optional[str] = None, overwrite: bool = True):
        """Save the current model checkpoint and remember the output directory.

        Args:
            dir_path: Optional output directory. Uses ``save_model_path`` when
                omitted.
            overwrite: Whether to overwrite an existing checkpoint directory.

        Returns:
            The directory path used for saving.
        """
        if self.model is None:
            raise RuntimeError("No model initialized.")
        out = dir_path or self.save_model_path
        self.model.save(out, overwrite=overwrite)
        metadata = self._build_checkpoint_metadata()
        self._write_checkpoint_metadata(out, metadata)
        # print(
        #     f"[save] wrote checkpoint metadata batch_names={metadata['batch_names']} "
        #     f"mods={metadata['mods']} n_obs={metadata['n_obs']}"
        # )
        self.prev_model_dir = out
        return out

    def get_latent_representation(self, mdata: Optional[mu.MuData] = None, *, kind: str = "joint"):
        """Extract latent representations from the active model.

        Args:
            mdata: Optional MuData object to encode. When omitted, MIDAS uses
                the data bound to the active model.
            kind: Latent representation kind passed through to MIDAS.

        Returns:
            A NumPy array containing latent embeddings.
        """
        if self.model is None:
            raise RuntimeError("No model initialized.")
        return self.model.get_latent_representation(mdata=mdata, kind=kind, verbose=False)

    @classmethod
    def load_model_from_checkpoint_for_mdata(
        cls,
        mdata: mu.MuData,
        model_dir: Path | str,
        *,
        batch_size: int = 256,
        batch_key: str = "batch",
    ):
        """Load a checkpoint into a fresh helper bound to a specific MuData object.

        Args:
            mdata: MuData object whose feature space should match the checkpoint.
            model_dir: Checkpoint directory containing ``model.pt``.
            batch_size: Batch size used for subsequent inference.
            batch_key: Observation column used as the batch identifier.

        Returns:
            A ``MIRACLE`` helper with a loaded MIDAS model.
        """
        ckpt_path, ckpt_dir = cls._resolve_checkpoint_path(model_dir)
        helper = cls(batch_size=batch_size, batch_key=batch_key)
        dims_x = mdata.uns.get("feat_dims")
        _MIDAS.setup_mudata(mdata, batch_key=batch_key, dims_x=dims_x)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model = _MIDAS(
            mdata,
            configs=ckpt.get("configs", helper.configs),
            batch_size=batch_size,
            save_model_path=str(ckpt_dir),
        )
        helper._transfer_compatible_state(model.net, ckpt["net"], module_name="net")
        helper._transfer_compatible_state(model.dsc, ckpt["dsc"], module_name="dsc")
        helper.model = model
        return helper

    @classmethod
    def get_latent_from_checkpoint(
        cls,
        mdata: mu.MuData,
        model_dir: Path | str,
        *,
        kind: str = "joint",
        batch_size: int = 128,
        batch_key: str = "batch",
    ) -> np.ndarray:
        """Load a checkpoint and immediately compute latent representations.

        Args:
            mdata: MuData object to encode.
            model_dir: Checkpoint directory containing ``model.pt``.
            kind: Latent representation kind passed through to MIDAS.
            batch_size: Batch size used for inference.
            batch_key: Observation column used as the batch identifier.

        Returns:
            A NumPy array containing latent embeddings.
        """
        helper = cls.load_model_from_checkpoint_for_mdata(
            mdata,
            model_dir,
            batch_size=batch_size,
            batch_key=batch_key,
        )
        return helper.get_latent_representation(mdata, kind=kind)

    def build_replay(
        self,
        replay_size: int = 2000,
        source_mdata: Optional[mu.MuData] = None,
        latent: Optional[np.ndarray] = None,
        *,
        strategy: Literal["subsample", "full"] = "subsample",
    ):
        """Build a replay MuData object from the current source data.

        Args:
            replay_size: Target number of replay observations to keep.
            source_mdata: Source MuData used to build replay. Defaults to the
                current task data bound to the helper.
            latent: Optional latent representation aligned to ``source_mdata``.
                Required only to avoid recomputing embeddings for subsampling.
            strategy: Replay construction strategy. ``subsample`` selects a
                compact subset, while ``full`` keeps all observations.

        Returns:
            A MuData object representing replay memory.
        """
        self.replay_size = int(replay_size)
        if source_mdata is None:
            source_mdata = self.current_mdata
        if source_mdata is None:
            raise RuntimeError("No source MuData available for replay.")
        if strategy not in {"subsample", "full"}:
            raise ValueError(f"Unsupported replay strategy: {strategy}")
        self.last_replay_source = strategy
        if strategy == "full":
            replay = self._subset_training_source(source_mdata, source_mdata.obs_names)
            self.attach_replay_metadata(
                replay,
                {
                    "batch_cell_counts_orig": self._get_training_cell_count_map_orig(
                        source_mdata,
                        self.replay_mdata if source_mdata is self.model._mdata else None,
                    )
                },
            )
            self.replay_mdata = replay
            return replay
        if latent is None:
            latent = self.get_latent_representation(source_mdata, kind="joint")
        obs_names = np.asarray(source_mdata.obs_names)
        batch_series = self._get_obs_batch_series(source_mdata)
        unique_batches = sorted(batch_series.unique().tolist())
        batch_masks = {batch: (batch_series == batch).to_numpy() for batch in unique_batches}
        batch_sizes = {batch: int(mask.sum()) for batch, mask in batch_masks.items()}

        total_cells = sum(batch_sizes.values())
        target_total = min(self.replay_size, total_cells)
        if target_total == total_cells:
            chosen = np.arange(total_cells)
        else:
            rates = {
                batch: int(batch_sizes[batch] / total_cells * target_total)
                for batch in unique_batches
            }
            remainder = target_total - sum(rates.values())
            if remainder > 0:
                leftovers = sorted(
                    unique_batches,
                    key=lambda batch: (
                        batch_sizes[batch] / total_cells * target_total - rates[batch],
                        batch_sizes[batch],
                    ),
                    reverse=True,
                )
                for batch in leftovers[:remainder]:
                    rates[batch] += 1

            chosen_parts = []
            for batch in unique_batches:
                batch_idx = np.flatnonzero(batch_masks[batch])
                target_size = min(rates[batch], len(batch_idx))
                if target_size == 0:
                    continue
                local_chosen = self._balltree_subsample(latent[batch_idx], target_size)
                chosen_parts.append(batch_idx[np.asarray(local_chosen, dtype=int)])
            chosen = np.sort(np.concatenate(chosen_parts)) if chosen_parts else np.array([], dtype=int)

        replay = self._subset_training_source(source_mdata, obs_names[chosen])
        self.attach_replay_metadata(
            replay,
            {
                "batch_cell_counts_orig": self._get_training_cell_count_map_orig(
                    source_mdata,
                    self.replay_mdata if source_mdata is self.model._mdata else None,
                )
            },
        )
        self.replay_mdata = replay
        return replay

    def _subset_training_source(self, source_mdata: mu.MuData, obs_names) -> mu.MuData:
        """Materialize selected rows from lazy continual sources when necessary."""
        is_lazy_view = (
            self._lazy_continual
            and self.model is not None
            and source_mdata is self.model._mdata
            and self._lazy_training_sources is not None
        )
        if not is_lazy_view:
            return self._subset_mudata(source_mdata, obs_names)

        requested = pd.Index(obs_names)
        parts = []
        for original in self._lazy_training_sources:
            selected = requested[requested.isin(original.obs_names)]
            if len(selected) > 0:
                parts.append(self._subset_mudata(original, selected))
        if not parts:
            raise ValueError("None of the requested observations exist in the lazy training sources.")
        return parts[0] if len(parts) == 1 else self._concat_mudata(parts)

    @staticmethod
    def load_replay(path: str) -> mu.MuData:
        """Load a replay MuData file from disk.

        Args:
            path: Path to a replay ``.h5mu`` file.

        Returns:
            The loaded MuData object.
        """
        return mu.read_h5mu(path)

    @staticmethod
    def attach_replay_metadata(mdata: mu.MuData, metadata: Dict):
        """Attach replay-specific metadata under the private MIRACLE namespace.

        Args:
            mdata: MuData object to annotate.
            metadata: Metadata mapping to merge into the replay namespace.

        Returns:
            The same MuData object with updated replay metadata.
        """
        current = copy.deepcopy(mdata.uns.get("_miracle_replay_meta", {}))
        current.update(copy.deepcopy(metadata))
        mdata.uns["_miracle_replay_meta"] = current
        return mdata

    def _build_checkpoint_metadata(self) -> Dict:
        """Build checkpoint metadata from the currently bound training MuData."""
        if self.model is None or getattr(self.model, "_mdata", None) is None:
            raise RuntimeError("No model-bound MuData available for checkpoint metadata export.")

        mdata = self.model._mdata
        replay_source = self.replay_mdata if self.mode == "continual" else None
        batch_names = list(map(str, getattr(self.model, "batch_names", self._get_batch_names(mdata))))
        batch_key = mdata.uns.get("_scmidas", {}).get("batch_key", self.batch_key)
        feat_dims = copy.deepcopy(mdata.uns.get("feat_dims", getattr(self.model, "dims_x", {})))
        feature_names = {
            mod: mdata.mod[mod].var_names.astype(str).tolist()
            for mod in mdata.mod.keys()
        }
        batch_cell_counts_orig = self._get_training_cell_count_map_orig(mdata, replay_source)
        metadata = {
            "batch_names": batch_names,
            "batch_key": str(batch_key),
            "feat_dims": feat_dims,
            "feature_names": feature_names,
            "mods": list(map(str, mdata.mod.keys())),
            "n_obs": int(mdata.n_obs),
            "batch_cell_counts_orig": batch_cell_counts_orig,
        }
        return metadata

    @staticmethod
    def _write_checkpoint_metadata(model_dir: Path | str, metadata: Dict) -> None:
        """Write MIRACLE metadata into an existing model.pt checkpoint."""
        model_dir = Path(model_dir)
        ckpt_path = model_dir / "model.pt"
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        ckpt["metadata"] = copy.deepcopy(metadata)
        torch.save(ckpt, ckpt_path)

    @staticmethod
    def load_replay_with_metadata(path: str, meta_path: Optional[str] = None) -> mu.MuData:
        """Load replay data and merge JSON metadata when available.

        Args:
            path: Path to a replay ``.h5mu`` file.
            meta_path: Optional JSON metadata path. Defaults to the same stem as
                ``path`` with a ``.json`` suffix.

        Returns:
            The loaded MuData object with replay metadata attached when found.
        """
        replay = mu.read_h5mu(path)
        meta_file = Path(meta_path) if meta_path is not None else Path(path).with_suffix(".json")
        if meta_file.exists():
            with open(meta_file) as f:
                MIRACLE.attach_replay_metadata(replay, json.load(f))
        return replay

    @staticmethod
    def export_mudata_metadata(mdata: mu.MuData, path: str, *, replay_strategy: str):
        """Export compact metadata needed to reconstruct replay context later.

        Args:
            mdata: MuData object whose metadata should be exported.
            path: Output JSON path.
            replay_strategy: Replay construction strategy label to store.

        Returns:
            The metadata dictionary that was written to disk.
        """
        batch_key = MIRACLE.batch_key_static()
        helper = MIRACLE(batch_key=batch_key)
        batch_series = helper._get_obs_batch_series(mdata)
        batch_names = sorted(set(batch_series.tolist()))
        batch_cell_counts = {
            batch_name: int((batch_series == batch_name).sum())
            for batch_name in batch_names
        }

        metadata = {
            "replay_strategy": replay_strategy,
            "mods": list(mdata.mod.keys()),
            "n_obs": int(mdata.n_obs),
            "batch_key": batch_key,
            "batch_names": batch_names,
            "batch_cell_counts": batch_cell_counts,
            "batch_cell_counts_orig": copy.deepcopy(
                mdata.uns.get("_miracle_replay_meta", {}).get("batch_cell_counts_orig", batch_cell_counts)
            ),
            "feat_dims": copy.deepcopy(mdata.uns.get("feat_dims", {})),
            "feature_names": {
                mod: mdata.mod[mod].var_names.astype(str).tolist()
                for mod in mdata.mod.keys()
            },
        }
        with open(path, "w") as f:
            json.dump(MIRACLE._to_jsonable(metadata), f, indent=2)
        return metadata

    @staticmethod
    def batch_key_static() -> str:
        """Return the default batch annotation key used by this wrapper.

        Args:
            None.

        Returns:
            The default batch observation key.
        """
        return "batch"

    @staticmethod
    def _concat_mudata(parts: Sequence[mu.MuData]) -> mu.MuData:
        """Concatenate multiple MuData objects across observations with feature alignment.

        Args:
            parts: Sequence of MuData objects to concatenate.

        Returns:
            A single concatenated MuData object with aligned feature ordering.
        """
        def _feature_key(name: str) -> str:
            if "-" in name:
                return name.split("-", 1)[0]
            return name

        def _build_target_layout(adatas: Sequence[ad.AnnData], ref_dims: Optional[Sequence[int]]):
            ref_vars = adatas[0].var_names.astype(str).tolist()
            if ref_dims is not None and len(ref_dims) > 1:
                if int(sum(ref_dims)) != len(ref_vars):
                    raise ValueError(
                        f"Reference feat_dims sum {sum(ref_dims)} does not match n_vars={len(ref_vars)}."
                    )
                segment_keys = []
                segment_ref_vars = []
                start = 0
                for dim in ref_dims:
                    end = start + int(dim)
                    seg_vars = ref_vars[start:end]
                    if not seg_vars:
                        raise ValueError("Encountered empty segmented feature block.")
                    segment_keys.append(_feature_key(seg_vars[0]))
                    segment_ref_vars.append(seg_vars)
                    start = end

                if len(set(segment_keys)) != len(segment_keys):
                    raise ValueError(
                        "Segmented feat_dims requires each feature block to have a distinct prefix "
                        f"before '-'; got segment keys {segment_keys}."
                    )

                extras_by_segment = {key: [] for key in segment_keys}
                seen = set(ref_vars)
                for adata_in in adatas[1:]:
                    for name in adata_in.var_names.astype(str):
                        if name in seen:
                            continue
                        key = _feature_key(name)
                        if key not in extras_by_segment:
                            raise ValueError(
                                f"New feature {name!r} maps to unknown segment {key!r}; "
                                "cannot update segmented feat_dims safely."
                            )
                        extras_by_segment[key].append(name)
                        seen.add(name)

                target_vars = []
                target_dims = []
                for key, seg_vars in zip(segment_keys, segment_ref_vars):
                    combined = seg_vars + extras_by_segment[key]
                    target_vars.extend(combined)
                    target_dims.append(len(combined))
                return target_vars, target_dims

            seen = set(ref_vars)
            new_vars = []
            for adata_in in adatas[1:]:
                for name in adata_in.var_names.astype(str):
                    if name not in seen:
                        new_vars.append(name)
                        seen.add(name)
            target_vars = ref_vars + new_vars
            return target_vars, [len(target_vars)]

        def _align_mask_vector(mask_value, source_vars: Sequence[str], target_vars: Sequence[str]) -> np.ndarray:
            mask_arr = np.asarray(mask_value, dtype=np.float32).reshape(-1)
            if mask_arr.shape[0] != len(source_vars):
                raise ValueError(
                    f"Mask length {mask_arr.shape[0]} does not match source vars {len(source_vars)}."
                )
            aligned = np.zeros(len(target_vars), dtype=np.float32)
            source_index = pd.Index(source_vars, dtype=str)
            positions = source_index.get_indexer(pd.Index(target_vars, dtype=str))
            present = positions >= 0
            aligned[present] = mask_arr[positions[present]]
            return aligned

        def _align_matrix_columns(X, source_vars: Sequence[str], target_vars: Sequence[str]):
            """Align columns without copying sparse values before the final concat."""
            target_pos = {name: i for i, name in enumerate(target_vars)}
            source_to_target = np.fromiter(
                (target_pos[name] for name in source_vars),
                dtype=np.int64,
                count=len(source_vars),
            )

            if issparse(X):
                X_csr = X.tocsr(copy=False)
                mapped_indices = source_to_target[X_csr.indices].astype(X_csr.indices.dtype, copy=False)
                return sp.csr_matrix(
                    (X_csr.data, mapped_indices, X_csr.indptr),
                    shape=(X_csr.shape[0], len(target_vars)),
                    copy=False,
                )

            X_array = np.asarray(X)
            aligned = np.zeros((X_array.shape[0], len(target_vars)), dtype=X_array.dtype)
            aligned[:, source_to_target] = X_array
            return aligned

        mods = sorted(set.union(*[set(p.mod.keys()) for p in parts])) if parts else []
        mod_map = {}
        merged_feat_dims = {}
        for mod in mods:
            adatas = [p.mod[mod] for p in parts if mod in p.mod]
            ref_dims = parts[0].uns.get("feat_dims", {}).get(mod)
            target_vars, target_dims = _build_target_layout(adatas, ref_dims)
            merged_feat_dims[mod] = list(map(int, target_dims))

            aligned_matrices = []
            aligned_obs = []
            merged_uns = {}
            target_index = pd.Index(target_vars, dtype=str)
            for adata_in in adatas:
                source_vars = adata_in.var_names.astype(str).tolist()
                if source_vars == target_vars:
                    X_aligned = adata_in.X
                else:
                    X_aligned = _align_matrix_columns(adata_in.X, source_vars, target_vars)

                for uns_key, uns_value in adata_in.uns.items():
                    if uns_key.startswith("mask_"):
                        merged_uns[uns_key] = _align_mask_vector(uns_value, source_vars, target_vars)

                aligned_matrices.append(X_aligned)
                aligned_obs.append(adata_in.obs.copy())

            if len(aligned_matrices) == 1:
                merged_X = aligned_matrices[0]
            elif any(issparse(X) for X in aligned_matrices):
                matrices = [X if issparse(X) else sp.csr_matrix(X) for X in aligned_matrices]
                merged_X = sp.vstack(matrices, format="csr")
            else:
                merged_X = np.concatenate([np.asarray(X) for X in aligned_matrices], axis=0)

            merged_obs = pd.concat(aligned_obs, axis=0, sort=False)
            merged_var = adatas[0].var.reindex(target_index).copy()
            for adata_in in adatas[1:]:
                incoming_var = adata_in.var.reindex(target_index)
                for column in incoming_var.columns:
                    if column not in merged_var.columns:
                        merged_var[column] = incoming_var[column]
                    else:
                        merged_var[column] = merged_var[column].where(
                            merged_var[column].notna(), incoming_var[column]
                        )

            merged = ad.AnnData(X=merged_X, obs=merged_obs, var=merged_var)
            if merged_uns:
                merged.uns.update(merged_uns)
            merged_vars = merged.var_names.astype(str)
            same_order = (
                len(target_vars) == len(merged_vars)
                and np.array_equal(np.asarray(merged_vars), np.asarray(target_vars, dtype=str))
            )
            if not same_order:
                raise ValueError(
                    f"_concat_mudata produced unexpected aligned var order for modality {mod!r}."
                )
            mod_map[mod] = merged
        out = mu.MuData(mod_map)
        feat_dims = copy.deepcopy(parts[0].uns.get("feat_dims", {})) if parts else {}
        feat_dims.update(merged_feat_dims)
        if feat_dims:
            out.uns['feat_dims'] = feat_dims
        out.update()
        return out

    @staticmethod
    def _subset_mudata(mdata: mu.MuData, obs_names) -> mu.MuData:
        """Subset a MuData object by observation names across all available modalities.

        Args:
            mdata: Input MuData object.
            obs_names: Observation names to keep.

        Returns:
            A subsetted MuData object.
        """
        obs_names = list(obs_names)
        mod_map = {}
        for mod, adata in mdata.mod.items():
            keep = [x for x in obs_names if x in adata.obs_names]
            if keep:
                view = adata[keep]
                mod_map[mod] = view.to_memory() if adata.isbacked else view.copy()
        out = mu.MuData(mod_map)
        if 'feat_dims' in mdata.uns:
            out.uns['feat_dims'] = copy.deepcopy(mdata.uns['feat_dims'])
        out.update()
        return out

    @staticmethod
    def _balltree_subsample(X: np.ndarray, target_size: int, leaf_size: int = 10):
        """Subsample latent points with a BallTree-based coverage heuristic.

        Args:
            X: Latent matrix of shape ``(n_obs, n_dim)``.
            target_size: Target number of observations to keep.
            leaf_size: BallTree leaf size used during subsampling.

        Returns:
            A sorted list of selected row indices.
        """
        n_obs = len(X)
        if target_size <= 0 or n_obs == 0:
            return []
        if target_size >= n_obs:
            return list(range(n_obs))
        if n_obs <= leaf_size:
            return sorted(np.random.choice(n_obs, size=target_size, replace=False).tolist())
        tree = BallTree(X, leaf_size=leaf_size)
        layer = int(np.log2(max(1, n_obs // leaf_size)))
        t = [1]
        for i in range(layer + 1):
            t.append(t[i] * 2)
        t = [i - 1 for i in t]
        t.sort(reverse=True)
        nodes = tree.get_arrays()[2]
        order = tree.get_arrays()[1]
        target = []
        target_set = set()
        for l in range(layer):
            layer_nodes = nodes[t[l + 1]:t[l]]
            if len(target) >= target_size:
                break
            n_nodes = len(layer_nodes)
            remain = target_size - len(target)
            s = remain // n_nodes
            if s == 0:
                continue
            for node in layer_nodes:
                start_id = node[0]
                end_id = node[1]
                candidates = [idx for idx in order[start_id:end_id] if idx not in target_set]
                if len(candidates) == 0:
                    continue
                if len(candidates) <= s:
                    chosen = candidates
                else:
                    chosen = list(np.random.choice(candidates, size=s, replace=False))
                target.extend(chosen)
                target_set.update(chosen)
        target = sorted(set(target))
        if len(target) < min(target_size, len(X)):
            remain = sorted(set(range(len(X))) - set(target))
            need = min(target_size, len(X)) - len(target)
            target.extend(remain[:need])
        return sorted(target[:target_size])

    @staticmethod
    def _transfer_compatible_state(module, source_state: Dict[str, torch.Tensor], module_name: str = "module"):
        """Copy compatible tensors from a checkpoint, allowing partial shape overlap.

        Args:
            module: Target PyTorch module to update.
            source_state: Source state dictionary loaded from a checkpoint.
            module_name: Friendly module name used in the summary printout.

        Returns:
            None.
        """
        def infer_source_dims_h(target_dims_h: Dict[str, List[int]]) -> Dict[str, int]:
            """Infer previous per-modality hidden widths from checkpoint tensors."""
            source_dims_h = {}
            for modality in target_dims_h.keys():
                transform_key = f"encoder.transform_concat.{modality}.net.0.weight"
                if transform_key in source_state:
                    source_dims_h[modality] = int(source_state[transform_key].shape[0])
                    continue

                encoder_key = f"encoder.encoders.{modality}.0.net.0.weight"
                if encoder_key in source_state:
                    source_dims_h[modality] = int(source_state[encoder_key].shape[1])
            return source_dims_h

        def infer_source_mod_order() -> List[str]:
            """Infer source modality order from the checkpoint state dict."""
            mod_order = []
            for key in source_state.keys():
                if key.startswith("encoder.encoders."):
                    modality = key.split(".")[2]
                elif key.startswith("decoder.transform_concat."):
                    modality = key.split(".")[2]
                elif key.startswith("decoder.post_decoders."):
                    modality = key.split(".")[2]
                else:
                    continue
                if modality not in mod_order:
                    mod_order.append(modality)
            return mod_order

        def get_mod_offsets(mod_order: Sequence[str], dims_h_by_mod: Dict[str, int]) -> Dict[str, int]:
            """Compute row offsets for modality blocks in shared decoder output."""
            offsets = {}
            start = 0
            for modality in mod_order:
                if modality not in dims_h_by_mod:
                    continue
                offsets[modality] = start
                start += int(dims_h_by_mod[modality])
            return offsets

        def copy_decoder_output_blocks(
            target_tensor: torch.Tensor,
            source_tensor: torch.Tensor,
            target_offsets: Dict[str, int],
            target_dims_h_by_mod: Dict[str, int],
            source_offsets: Dict[str, int],
            source_dims_h: Dict[str, int],
        ) -> Optional[torch.Tensor]:
            """Copy shared-decoder output blocks modality by modality."""
            common_modalities = [m for m in target_offsets.keys() if m in source_offsets and m in source_dims_h]
            if not common_modalities:
                return None

            target_copy = target_tensor.clone()
            copied = False

            for modality in common_modalities:
                target_row_start = int(target_offsets[modality])
                source_row_start = int(source_offsets[modality])
                target_rows = int(target_dims_h_by_mod[modality])
                source_rows = int(source_dims_h[modality])
                row_count = min(source_rows, target_rows)
                if source_tensor.ndim == 2:
                    col_count = min(source_tensor.shape[1], target_copy.shape[1])
                    target_copy[
                        target_row_start:target_row_start + row_count,
                        :col_count,
                    ] = source_tensor[
                        source_row_start:source_row_start + row_count,
                        :col_count,
                    ]
                else:
                    target_copy[target_row_start:target_row_start + row_count] = source_tensor[
                        source_row_start:source_row_start + row_count
                    ]
                copied = True

            return target_copy if copied else None

        target_state = module.state_dict()
        target_dims_h = getattr(module, "dims_h", None)
        source_dims_h = (
            infer_source_dims_h(target_dims_h)
            if module_name == "net" and isinstance(target_dims_h, dict)
            else {}
        )
        target_hidden_total = (
            sum(int(dim[0]) for dim in target_dims_h.values())
            if isinstance(target_dims_h, dict)
            else None
        )
        target_dims_h_by_mod = (
            {modality: int(dim[0]) for modality, dim in target_dims_h.items()}
            if isinstance(target_dims_h, dict)
            else {}
        )
        source_hidden_total = sum(source_dims_h.values()) if source_dims_h else None
        source_mod_order = infer_source_mod_order() if source_dims_h else []
        target_mod_order = list(target_dims_h_by_mod.keys())
        source_offsets = get_mod_offsets(source_mod_order, source_dims_h) if source_dims_h else {}
        target_offsets = get_mod_offsets(target_mod_order, target_dims_h_by_mod) if target_dims_h_by_mod else {}
        if module_name == "net":
            # print(f"[transfer/net] source_mod_order={source_mod_order}")
            # print(f"[transfer/net] target_mod_order={target_mod_order}")
            # print(f"[transfer/net] source_dims_h={source_dims_h}")
            # print(f"[transfer/net] target_dims_h={target_dims_h_by_mod}")
            # print(f"[transfer/net] source_offsets={source_offsets}")
            # print(f"[transfer/net] target_offsets={target_offsets}")
            pass
        loaded = 0
        partial = 0
        skipped = []
        for key, target_tensor in target_state.items():
            if key not in source_state:
                skipped.append((key, "missing"))
                continue
            source_tensor = source_state[key]
            if source_tensor.shape == target_tensor.shape:
                target_state[key] = source_tensor
                loaded += 1
                continue
            if (
                module_name == "net"
                and key.startswith("decoder.shared_decoder.net.")
                and (key.endswith(".weight") or key.endswith(".bias"))
                and source_dims_h
                and target_hidden_total == int(target_tensor.shape[0])
                and source_hidden_total == int(source_tensor.shape[0])
            ):
                block_copy = copy_decoder_output_blocks(
                    target_tensor,
                    source_tensor,
                    target_offsets,
                    target_dims_h_by_mod,
                    source_offsets,
                    source_dims_h,
                )
                if block_copy is not None:
                    # print(
                    #     f"[transfer/{module_name}] block_copy key={key} "
                    #     f"source_shape={tuple(source_tensor.shape)} target_shape={tuple(target_tensor.shape)}"
                    # )
                    target_state[key] = block_copy
                    partial += 1
                    continue
            if source_tensor.ndim == target_tensor.ndim and source_tensor.ndim in (1, 2):
                slices = tuple(slice(0, min(s, t)) for s, t in zip(source_tensor.shape, target_tensor.shape))
                target_copy = target_tensor.clone()
                target_copy[slices] = source_tensor[slices]
                target_state[key] = target_copy
                partial += 1
                continue
            skipped.append((key, f"shape {tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"))
        module.load_state_dict(target_state)
        # print(f"[{module_name}] loaded={loaded}, partial={partial}, skipped={len(skipped)}")
        if skipped:
            preview = skipped[:10]
            # print(f"[{module_name}] skipped_preview={preview}")
