Quick Start
===========

This page describes the current MIRACLE API. It intentionally uses
``scmiracle.model.MIRACLE`` rather than legacy ``scmidas.model.MIDAS`` examples:
MIDAS remains the underlying model implementation, while MIRACLE provides the
continual integration workflow.

Imports
-------

.. code-block:: python

   import mudata as mu

   from scmiracle.model import MIRACLE
   from scmiracle.download_data import download, list_available_datasets

Optional example data can be downloaded with:

.. code-block:: python

   list_available_datasets()
   download("DOTEA_mudata", "./data")

Data Contract
-------------

MIRACLE expects a :class:`mudata.MuData` object with:

* one modality per ``mdata.mod`` entry, for example ``rna``, ``adt``, or
  ``atac``;
* a batch annotation column, named ``batch`` by default, available in top-level
  ``mdata.obs`` or modality-specific ``adata.obs``;
* optional ``mdata.uns["feat_dims"]`` for segmented feature spaces such as
  chromosome-level ATAC blocks;
* aligned observation names across modalities when cells are shared, and unique
  names when datasets are concatenated across tasks.

If your data is stored in the original MIRACLE mtx-directory layout, convert it
before training:

.. code-block:: python

   mdata = MIRACLE.load_mtx_dir_as_mudata(
       "path/to/miracle_mtx_dir",
       label_path="path/to/labels.csv",
       batch_prefix="step1",
   )

Initial Training
----------------

Train the first integration model from a MuData object:

.. code-block:: python

   mdata = mu.read_h5mu("./data/DOTEA_mudata/step1.h5mu")

   miracle = MIRACLE(
       batch_key="batch",
       batch_size=128,
       save_model_path="./saved_models/miracle_step1",
   )

   miracle.setup_mudata(mdata)
   miracle.train(
       max_epochs=2000,
       accelerator="auto",
       devices=1,
       strategy="auto",
   )

   latent = miracle.get_latent_representation(kind="joint")
   mdata.obsm["X_miracle"] = latent
   model_dir = miracle.save()

``setup_mudata`` registers MIDAS-compatible metadata and initializes the model.
``train`` forwards keyword arguments to the underlying MIDAS training routine,
which uses Lightning internally. ``save`` writes ``model.pt`` and embeds MIRACLE
metadata required by later continual steps.

Build Replay Memory
-------------------

After a step is trained, build a replay MuData object for future continual
integration:

.. code-block:: python

   replay = miracle.build_replay(
       replay_size=2000,
       source_mdata=mdata,
       latent=latent,
       strategy="subsample",
   )
   replay.write_h5mu("./saved_models/replay_step1.h5mu")
   MIRACLE.export_mudata_metadata(
       replay,
       "./saved_models/replay_step1.json",
       replay_strategy="subsample",
   )

``strategy="subsample"`` uses latent-space coverage to keep a compact replay
set. ``strategy="full"`` keeps all observations and is useful for small
datasets or validation runs.

Continual Integration
---------------------

For a new task or data release, combine the replay memory with the current data
and transfer compatible parameters from the previous checkpoint:

.. code-block:: python

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

``lazy=True`` avoids materializing a fully concatenated expression matrix and is
the recommended mode for large continual updates. Set ``lazy=False`` only when
you need the legacy eager merge behavior for debugging or compatibility.

Important Constraints
---------------------

* Current and replay batch names must be disjoint. Use
  ``MIRACLE.prefix_batch_and_obs_names(...)`` before setup if a reused batch
  label would collide.
* Replay metadata stores original per-batch cell counts. Preserve the JSON file
  produced by ``export_mudata_metadata`` when replay data is moved between
  machines.
* Feature alignment keeps reference features first and appends newly observed
  features. For segmented feature dimensions, segment prefixes before ``-`` must
  remain unambiguous.
* The active public import is ``from scmiracle.model import MIRACLE``.

Inference from a Checkpoint
---------------------------

Load a saved model and compute latents for a compatible MuData object:

.. code-block:: python

   z = MIRACLE.get_latent_from_checkpoint(
       mdata,
       "./saved_models/miracle_step2",
       kind="joint",
       batch_size=128,
       batch_key="batch",
   )

For repeated inference, keep the helper object:

.. code-block:: python

   helper = MIRACLE.load_model_from_checkpoint_for_mdata(
       mdata,
       "./saved_models/miracle_step2",
       batch_size=128,
   )
   mdata.obsm["X_miracle"] = helper.get_latent_representation(mdata)
