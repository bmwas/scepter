# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
from scepter.modules.utils.import_utils import LazyImportModule
"""
Normally, hooks have priorities, below we recommend priority that runs fine (low score MEANS high priority)
BackwardHook: 0
LogHook: 100
LrHook: 200
CheckpointHook: 300
SamplerHook: 400

Recommend sequences in training are:
before solve:
    TensorboardLogHook: prepare file handler
    WandbLogHook: prepare file handler
    CheckpointHook: resume checkpoint
    WandbCheckpointHook: track model config
    WandbProbeDataHook: prepare visualization tables
    WandbLrHook: prepare learning rate tracking
    WandbValLossHook: prepare validation loss tracking
    WandbFileTrackerHook: initialize file tracking

before epoch:
    LogHook: clear epoch variables
    DistSamplerHook: change sampler seed

before iter:
    LogHook: record data time

after iter:
    BackwardHook: network backward
    LogHook: log
    TensorboardLogHook: log
    WandbLogHook: log
    CheckpointHook: save checkpoint
    WandbCheckpointHook: track checkpoint artifacts
    WandbProbeDataHook: log visualization data
    WandbLrHook: log learning rate changes
    WandbValLossHook: log validation metrics
    WandbFileTrackerHook: periodically check for new files
    SafetensorsHook: save checkpoint

after epoch:
    LrHook: reset learning rate
    WandbLrHook: log learning rate updates
    CheckpointHook: save checkpoint
    WandbCheckpointHook: track epoch artifacts
    WandbProbeDataHook: log evaluation visualizations
    WandbValLossHook: log validation results
    WandbFileTrackerHook: track new result files

after solve:
    TensorboardLogHook: close file handler
    WandbLogHook: close file handler
    WandbCheckpointHook: final metrics
    WandbProbeDataHook: finalize visualization tables
    WandbValLossHook: finalize validation metrics
    WandbFileTrackerHook: final file tracking and artifact upload
"""


if TYPE_CHECKING:
    from scepter.modules.solver.hooks.backward import BackwardHook
    from scepter.modules.solver.hooks.checkpoint import CheckpointHook
    from scepter.modules.solver.hooks.data_probe import ProbeDataHook
    from scepter.modules.solver.hooks.ema import ModelEmaHook
    from scepter.modules.solver.hooks.hook import Hook
    from scepter.modules.solver.hooks.log import LogHook, TensorboardLogHook
    from scepter.modules.solver.hooks.lr import LrHook
    from scepter.modules.solver.hooks.registry import HOOKS
    from scepter.modules.solver.hooks.safetensors import SafetensorsHook
    from scepter.modules.solver.hooks.sampler import DistSamplerHook
    from scepter.modules.solver.hooks.val_loss import ValLossHook
    from scepter.modules.solver.hooks.wandb_log import WandbLogHook
    from scepter.modules.solver.hooks.wandb_checkpoint import WandbCheckpointHook
    from scepter.modules.solver.hooks.wandb_probe import WandbProbeDataHook
    from scepter.modules.solver.hooks.wandb_lr import WandbLrHook
    from scepter.modules.solver.hooks.wandb_val_loss import WandbValLossHook
    from scepter.modules.solver.hooks.wandb_file_tracker import WandbFileTrackerHook
    from scepter.modules.solver.hooks.wandb_dataset_artifact import WandbDatasetArtifactHook
else:
    _import_structure = {
        'backward': ['BackwardHook'],
        'checkpoint': ['CheckpointHook'],
        'data_probe': ['ProbeDataHook'],
        'ema': ['ModelEmaHook'],
        'hook': ['Hook'],
        'log': ['LogHook', 'TensorboardLogHook'],
        'lr': ['LrHook'],
        'registry': ['HOOKS'],
        'safetensors': ['SafetensorsHook'],
        'sampler': ['DistSamplerHook'],
        'val_loss': ['ValLossHook'],
        'wandb_log': ['WandbLogHook'],
        'wandb_checkpoint': ['WandbCheckpointHook'],
        'wandb_probe': ['WandbProbeDataHook'],
        'wandb_lr': ['WandbLrHook'],
        'wandb_val_loss': ['WandbValLossHook'],
        'wandb_file_tracker': ['WandbFileTrackerHook'],
        'wandb_dataset_artifact': ['WandbDatasetArtifactHook']
    }

    import sys
    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
