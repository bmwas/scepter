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
        },
        'EARLY_INIT': {
            'value': True, 
            'description': 'whether to initialize wandb early before model loading'
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
        self.early_init = cfg.get('EARLY_INIT', True)
        self._local_log_dir = None
        self.wandb_run = None
        
        # Load wandb API key from .env file
        load_dotenv()
        # Debug logging for WANDB_API_KEY
        api_key = os.getenv('WANDB_API_KEY')
        if not api_key:
            warnings.warn('WANDB_API_KEY not found in environment. Wandb tracking will not work!')
            if self.logger:
                self.logger.error('WANDB_API_KEY not found in environment. Wandb tracking will not work!')
            else:
                print('WANDB_API_KEY not found in environment. Wandb tracking will not work!')
        else:
            if self.logger:
                self.logger.info('WANDB_API_KEY successfully loaded.')
            else:
                print('WANDB_API_KEY successfully loaded.')
                
        # Debug: Check if wandb is imported correctly
        try:
            import wandb
            if self.logger:
                self.logger.info(f"Wandb version: {wandb.__version__}")
            else:
                print(f"Wandb version: {wandb.__version__}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error importing wandb: {e}")
            else:
                print(f"Error importing wandb: {e}")
        
        # If early initialization is enabled, we'll set up logdir in constructor
        if self.early_init and self.log_dir is not None:
            self._local_log_dir, _ = FS.map_to_local(self.log_dir)
            os.makedirs(self._local_log_dir, exist_ok=True)

    def init_wandb(self, solver=None, config=None):
        """Initialize wandb run early, before model loading.
        """
        if we.rank != 0:
            return
            
        # Debug: Print initialization attempt
        if solver and hasattr(solver, 'logger'):
            solver.logger.info("===== ATTEMPTING TO INITIALIZE WANDB IN init_wandb() =====")
        else:
            print("===== ATTEMPTING TO INITIALIZE WANDB IN init_wandb() =====")
            
        # Check if wandb is already initialized
        import wandb
        current_run = wandb.run
        
        if current_run is not None:
            # Wandb is already initialized by BaseSolver, use that run
            self.wandb_run = current_run
            if solver and hasattr(solver, 'logger'):
                solver.logger.info(f"Using existing wandb run: {self.wandb_run.name} (URL: {self.wandb_run.url})")
            else:
                print(f"Using existing wandb run: {self.wandb_run.name} (URL: {self.wandb_run.url})")
                
            # Update run config with our settings
            if self.project_name:
                self.wandb_run.project = self.project_name
            if self.run_name:
                self.wandb_run.name = self.run_name
            if self.tags:
                self.wandb_run.tags = self.wandb_run.tags + tuple(self.tags)
            if config:
                self.wandb_run.config.update(config)
                
            return
            
        # If wandb_run is already set, skip initialization
        if self.wandb_run is not None:
            if solver and hasattr(solver, 'logger'):
                solver.logger.info("Wandb already initialized, skipping initialization.")
            else:
                print("Wandb already initialized, skipping initialization.")
            return
            
        # Set up log directory if not already done
        if self._local_log_dir is None:
            if solver is not None and self.log_dir is None:
                self.log_dir = osp.join(solver.work_dir, 'wandb')
            if self.log_dir is not None:
                self._local_log_dir, _ = FS.map_to_local(self.log_dir)
                os.makedirs(self._local_log_dir, exist_ok=True)
        
        # Initialize wandb with minimal config initially
        try:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                dir=self._local_log_dir,
                config=config or {},
                save_code=self.save_code,
                tags=self.tags,
                entity=self.entity,
                resume="allow"
            )
            
            # Debug logging for wandb initialization
            if self.wandb_run is not None:
                if solver and hasattr(solver, 'logger'):
                    solver.logger.info(f'Wandb initialized: run name={self.wandb_run.name}, project={self.project_name}')
                    solver.logger.info(f'Wandb dashboard URL: {self.wandb_run.url}')
                else:
                    print(f'Wandb initialized: run name={self.wandb_run.name}, project={self.project_name}')
                    print(f'Wandb dashboard URL: {self.wandb_run.url}')
            else:
                if solver and hasattr(solver, 'logger'):
                    solver.logger.error('Wandb initialization failed - wandb.init() returned None.')
                else:
                    print('Wandb initialization failed - wandb.init() returned None.')
        except Exception as e:
            if solver and hasattr(solver, 'logger'):
                solver.logger.error(f'Error initializing wandb: {e}')
                import traceback
                solver.logger.error(traceback.format_exc())
            else:
                print(f'Error initializing wandb: {e}')
                import traceback
                print(traceback.format_exc())

    def before_solve(self, solver):
        if we.rank != 0:
            return

        if self.log_dir is None:
            self.log_dir = osp.join(solver.work_dir, 'wandb')

        if self._local_log_dir is None:
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
        
        # Initialize wandb run if not already initialized via init_wandb()
        if self.wandb_run is None:
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
            
            solver.logger.info(f'Wandb: initialized run {self.wandb_run.name} in project {self.project_name}')
            solver.logger.info(f'Wandb: dashboard available at {self.wandb_run.url}')
        else:
            # If already initialized, update config
            for k, v in wandb_config.items():
                self.wandb_run.config[k] = v
            
            solver.logger.info(f'Wandb: updated configuration for run {self.wandb_run.name}')
        
        # Log git information if available
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            git_config = {
                'git.commit': repo.head.commit.hexsha,
                'git.branch': repo.active_branch.name
            }
            for k, v in git_config.items():
                self.wandb_run.config[k] = v
        except Exception:
            pass
            
        # Save initial model architecture as artifact
        if hasattr(solver, 'model'):
            try:
                model_artifact = wandb.Artifact(
                    name=f"model_architecture_{self.wandb_run.id}", 
                    type="model_architecture"
                )
                
                # Save model summary if possible
                if hasattr(solver.model, 'cfg'):
                    model_config = solver.model.cfg
                    model_artifact.metadata.update(self._flatten_config(model_config))
                
                # Add model parameters count
                if hasattr(solver, '_model_parameters') and solver._model_parameters > 0:
                    model_artifact.metadata["parameters_count"] = solver._model_parameters
                    
                # Add model FLOPS if available
                if hasattr(solver, '_model_flops') and solver._model_flops > 0:
                    model_artifact.metadata["flops"] = solver._model_flops
                
                self.wandb_run.log_artifact(model_artifact)
            except Exception as e:
                solver.logger.warning(f"Error logging model architecture: {e}")

        self.start_time = time.time()
        self.count: defaultdict = defaultdict(int)
        self.max_val: defaultdict = defaultdict(int)
        self.min_val: defaultdict = defaultdict(int)
        self.sum_val: defaultdict = defaultdict(int)
        self.avg_val: defaultdict = defaultdict(int)

    def after_iter(self, solver):
        """Called after each iteration.
        
        This method logs metrics to wandb in the same way TensorboardLogHook does.
        It ensures wandb and tensorboard have the same metrics logged.
        """
        if we.rank != 0 or self.wandb_run is None:
            # Debug: Log why we're not logging
            if solver.total_iter % 100 == 0:  # Only log every 100 iterations to avoid spam
                if we.rank != 0:
                    solver.logger.info("WandbLogHook.after_iter: Not logging because we.rank != 0")
                elif self.wandb_run is None:
                    solver.logger.error("WandbLogHook.after_iter: Not logging because self.wandb_run is None!")
            return
            
        # Debug: Log that we're attempting to log metrics
        if solver.total_iter % 100 == 0:  # Only log every 100 iterations to avoid spam
            solver.logger.info(f"WandbLogHook.after_iter: Logging metrics at iteration {solver.total_iter}")
            
        outputs = solver.iter_outputs.copy()
        extra_vars = solver.collect_log_vars()
        outputs.update(extra_vars)
        mode = solver.mode
        
        # Create a dict of metrics to log to wandb
        wandb_metrics = {}
        
        # Process all metrics from outputs exactly like TensorboardLogHook does
        for key, value in outputs.items():
            if key == 'batch_size':
                continue
            if isinstance(value, torch.Tensor):
                # Must be scalar
                if not value.ndim == 0:
                    # Handle images if present
                    if key.endswith('_img') and value.ndim == 4:
                        try:
                            # Add image data
                            if value.shape[1] in [1, 3]:  # Check if channels are in correct position
                                wandb_metrics[f'{mode}/images/{key}'] = wandb.Image(
                                    value[0].detach().cpu().float().numpy().transpose(1, 2, 0),
                                    caption=key
                                )
                        except Exception:
                            pass
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
                continue
                
            # Use same naming convention as TensorboardLogHook
            wandb_metrics[f'{mode}/iter/{key}'] = value
            
        # Add step information
        wandb_metrics['global_step'] = solver.total_iter
        
        # Log learning rate if available
        if hasattr(solver, 'optimizer'):
            for i, param_group in enumerate(solver.optimizer.param_groups):
                if 'lr' in param_group:
                    wandb_metrics[f'{mode}/iter/lr_group_{i}'] = param_group['lr']
        
        # Log to wandb
        try:
            self.wandb_run.log(wandb_metrics, step=solver.total_iter)
            if solver.total_iter % 100 == 0:  # Only log every 100 iterations to avoid spam
                solver.logger.info(f"WandbLogHook: Successfully logged {len(wandb_metrics)} metrics to wandb")
        except Exception as e:
            solver.logger.error(f"Error logging to wandb: {e}")
            import traceback
            solver.logger.error(traceback.format_exc())
        
        # Sync to wandb server at the same interval as TensorboardLogHook flushes
        if solver.total_iter % self.interval == 0:
            try:
                # Force sync
                self.wandb_run.log({}, commit=True)
                # Put to remote file systems every epoch
                FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)
                solver.logger.info(f"WandbLogHook: Synced wandb at iteration {solver.total_iter}")
            except Exception as e:
                solver.logger.error(f"Error syncing wandb: {e}")
                import traceback
                solver.logger.error(traceback.format_exc())

    def after_epoch(self, solver):
        """Called after each epoch.
        
        This method logs epoch metrics to wandb in the same way TensorboardLogHook does.
        """
        if we.rank != 0 or self.wandb_run is None:
            return
            
        outputs = solver.epoch_outputs.copy()
        
        # Create a dict of metrics to log to wandb
        wandb_metrics = {}
        
        # Process metrics the same way TensorboardLogHook does
        for mode, kvs in outputs.items():
            for key, value in kvs.items():
                wandb_metrics[f'{mode}/epoch/{key}'] = value
                
        # Log to wandb
        self.wandb_run.log(wandb_metrics, step=solver.epoch)
        
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
