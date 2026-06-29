# Release Notes

## 0.1.0

Initial `scmiracle` package release focused on continual integration for
single-cell multimodal data.

### API

- Public high-level helper: `scmiracle.model.MIRACLE`.
- Initial training: `MIRACLE.setup_mudata(...).train(...)`.
- Continual training: `MIRACLE.setup_continual(...)` with replay memory and a
  previous checkpoint.
- Inference: `get_latent_representation`,
  `load_model_from_checkpoint_for_mdata`, and `get_latent_from_checkpoint`.
- Replay utilities: `build_replay`, `load_replay`, `load_replay_with_metadata`,
  `attach_replay_metadata`, and `export_mudata_metadata`.
- Data migration utilities: `load_mtx_dir_as_mudata`,
  `prefix_batch_and_obs_names`, `attach_feature_and_label_metadata`, and
  `get_obs_series`.

### Continual Integration

- Replay-aware training alternates current-task and replay batches.
- Original per-batch cell counts are stored in checkpoint and replay metadata
  for stable weighting across incremental updates.
- Compatible checkpoint weights are transferred into expanded feature and batch
  spaces when possible.
- Lazy continual setup aligns features without materializing a full merged
  expression matrix.

### API Check Status

- `from scmiracle.model import MIRACLE` imports successfully in the current
  workspace.
- `scmiracle.download_data.download` and `list_available_datasets` are
  importable.
- `scmiracle.MIRACLE` is not currently available because `scmiracle/__init__.py`
  does not export the class.
