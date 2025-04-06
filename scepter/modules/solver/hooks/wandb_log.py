# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import os
import os.path as osp
import time
import warnings
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np
import torch
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import LogAgg, time_since

try:
    import wandb
    from dotenv import load_dotenv
except Exception as e:
    warnings.warn(f'Running without wandb! {e}')

_DEFAULT_LOG_PRIORITY = 100


@HOOKS.register_class()
class WandbLogHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LOG_PRIORITY,
            'description': 'the priority for processing!'
        },
        'LOG_DIR': {
            'value': None,
            'description': 'the dir for wandb log!'
        },
        'PROJECT_NAME': {
            'value': 'scepter-project',
            'description': 'the name of the wandb project!'
        },
        'RUN_NAME': {
            'value': None,
            'description': 'the name of the wandb run!'
        },
        'LOG_INTERVAL': {
            'value': 10000,
            'description': 'the interval for log upload!'
        },
        'CONFIG_LOGGING': {
            'value': True,
            'description': 'whether to log the configuration to wandb!'
        },
        'SAVE_CODE': {
            'value': True,
            'description': 'whether to save the code to wandb!'
        },
        'TAGS': {
            'value': [],
            'description': 'tags for the wandb run!'
        },
        'ENTITY': {
            'value': None,
            'description': 'entity (team) for the wandb run!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbLogHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_LOG_PRIORITY)
        self.log_dir = cfg.get('LOG_DIR', None)
        self.project_name = cfg.get('PROJECT_NAME', 'scepter-project')
        self.run_name = cfg.get('RUN_NAME', None)
        self.log_interval = cfg.get('LOG_INTERVAL', 1000)
        self.interval = cfg.get('INTERVAL', self.log_interval)
        self.config_logging = cfg.get('CONFIG_LOGGING', True)
        self.save_code = cfg.get('SAVE_CODE', True)
        self.tags = cfg.get('TAGS', [])
        self.entity = cfg.get('ENTITY', None)
        self._local_log_dir = None
        self.wandb_run = None
        
        # Load wandb API key from .env file
        load_dotenv()

    def before_solve(self, solver):
        if we.rank != 0:
            return

        if self.log_dir is None:
            self.log_dir = osp.join(solver.work_dir, 'wandb')

        self._local_log_dir, _ = FS.map_to_local(self.log_dir)
        os.makedirs(self._local_log_dir, exist_ok=True)
        
        # Set up wandb configuration
        wandb_config = {}
        
        # Get important configurations from solver
        if self.config_logging and hasattr(solver, 'cfg'):
            # Add solver config
            wandb_config.update(self._flatten_config(solver.cfg))
            
            # Add model config if available
            if hasattr(solver, 'model') and hasattr(solver.model, 'cfg'):
                wandb_config.update(self._flatten_config(solver.model.cfg, prefix='model'))
            
            # Add optimizer config if available
            if hasattr(solver, 'optimizer') and hasattr(solver.optimizer, 'defaults'):
                for k, v in solver.optimizer.defaults.items():
                    if isinstance(v, (int, float, str, bool)):
                        wandb_config[f'optimizer.{k}'] = v
            
            # Add dataset config if available
            if hasattr(solver, 'datas'):
                for data_key, data_obj in solver.datas.items():
                    if hasattr(data_obj, 'cfg'):
                        wandb_config.update(self._flatten_config(data_obj.cfg, prefix=f'data.{data_key}'))
        
        # Initialize wandb run
        self.wandb_run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            dir=self._local_log_dir,
            config=wandb_config,
            save_code=self.save_code,
            tags=self.tags,
            entity=self.entity,
            resume="allow"
        )
        
        # Log git information if available
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            wandb_config.update({
                'git.commit': repo.head.commit.hexsha,
                'git.branch': repo.active_branch.name
            })
        except Exception:
            pass
            
        solver.logger.info(f'Wandb: initialized run {self.wandb_run.name} in project {self.project_name}')
        solver.logger.info(f'Wandb: dashboard available at {self.wandb_run.url}')
        
        # Save initial model architecture as artifact
        if hasattr(solver, 'model'):
            try:
                model_summary = self._get_model_summary(solver.model)
                # Create model architecture artifact
                model_artifact = wandb.Artifact(
                    name=f"model-architecture-{self.wandb_run.id}",
                    type="model-architecture",
                    description="Initial model architecture"
                )
                with model_artifact.new_file("model_summary.txt") as f:
                    f.write(model_summary)
                self.wandb_run.log_artifact(model_artifact)
            except Exception as e:
                solver.logger.warning(f"Failed to log model architecture: {e}")

    def after_iter(self, solver):
        if self.wandb_run is None:
            return
            
        outputs = solver.iter_outputs.copy()
        extra_vars = solver.collect_log_vars()
        outputs.update(extra_vars)
        mode = solver.mode
        
        # Create a dict to store metrics for this step
        log_dict = {}
        
        # Process metrics
        for key, value in outputs.items():
            if key == 'batch_size':
                continue
                
            if isinstance(value, torch.Tensor):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = value.item()
            elif isinstance(value, np.ndarray):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = float(value)
            elif isinstance(value, numbers.Number):
                # Must be number
                pass
            else:
                # Handle images if present
                if key.endswith('_img') and isinstance(value, torch.Tensor) and value.ndim == 4:
                    try:
                        # Add image data
                        if value.shape[1] in [1, 3]:  # Check if channels are in correct position
                            log_dict[f'{mode}/images/{key}'] = wandb.Image(
                                value[0].detach().cpu().float().numpy().transpose(1, 2, 0),
                                caption=key
                            )
                        continue
                    except Exception:
                        pass
                continue

            # Add scalar metrics with proper prefix
            log_dict[f'{mode}/iter/{key}'] = value
        
        # Log learning rate if available
        if hasattr(solver, 'optimizer'):
            for i, param_group in enumerate(solver.optimizer.param_groups):
                if 'lr' in param_group:
                    log_dict[f'{mode}/iter/lr_group_{i}'] = param_group['lr']
        
        # Log metrics to wandb
        if log_dict:
            self.wandb_run.log(log_dict, step=solver.total_iter)
            
        # Sync wandb at intervals
        if solver.total_iter % self.interval == 0:
            self.wandb_run.log({}, commit=True)  # Force sync
            # Put to remote file systems every epoch
            FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_epoch(self, solver):
        if self.wandb_run is None:
            return
            
        outputs = solver.epoch_outputs.copy()
        log_dict = {}
        
        # Log epoch metrics
        for mode, kvs in outputs.items():
            for key, value in kvs.items():
                log_dict[f'{mode}/epoch/{key}'] = value
        
        if log_dict:
            self.wandb_run.log(log_dict, step=solver.total_iter)
        
        # Save model checkpoint as artifact at the end of each epoch
        if hasattr(solver, 'model') and we.rank == 0:
            try:
                # Create checkpoint artifact
                checkpoint_artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{solver.epoch}",
                    type="model-checkpoint",
                    description=f"Model checkpoint at epoch {solver.epoch}"
                )
                
                # Add metadata to the artifact
                checkpoint_artifact.metadata = {
                    "epoch": solver.epoch,
                    "iteration": solver.total_iter,
                    "timestamp": time.time()
                }
                
                # Add performance metrics to metadata if available
                if outputs:
                    for mode, kvs in outputs.items():
                        for key, value in kvs.items():
                            if isinstance(value, (int, float)):
                                checkpoint_artifact.metadata[f"{mode}_{key}"] = value
                
                # Add checkpoint file to artifact if it exists
                checkpoint_path = osp.join(solver.work_dir, f"epoch_{solver.epoch}.pth")
                if osp.exists(checkpoint_path):
                    checkpoint_artifact.add_file(checkpoint_path)
                    self.wandb_run.log_artifact(checkpoint_artifact)
            except Exception as e:
                solver.logger.warning(f"Failed to log checkpoint artifact: {e}")
        
        # Sync wandb
        self.wandb_run.log({}, commit=True)  # Force sync
        
        # Put to remote file systems every epoch
        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_solve(self, solver):
        if self.wandb_run is None:
            return
            
        # Close wandb run
        self.wandb_run.finish()
        
        # Sync to remote filesystem
        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def _flatten_config(self, cfg: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested configuration for wandb logging."""
        result = {}
        for key, value in cfg.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested = self._flatten_config(value, f"{prefix}.{key}" if prefix else key)
                result.update(nested)
            elif isinstance(value, (int, float, str, bool, list)) or value is None:
                # Only include primitive types that wandb can handle
                full_key = f"{prefix}.{key}" if prefix else key
                result[full_key] = value
        return result

    def _get_model_summary(self, model) -> str:
        """Get a summary of the model architecture."""
        try:
            from torchinfo import summary
            model_summary = summary(model, depth=5, verbose=0)
            return str(model_summary)
        except ImportError:
            return str(model)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbLogHook.para_dict,
                            set_name=True)
