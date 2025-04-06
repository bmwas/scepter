# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import json
import time
import warnings
from typing import Optional, Dict, Any

import torch
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

try:
    import wandb
except Exception as e:
    warnings.warn(f'Running without wandb! {e}')

_DEFAULT_CHECKPOINT_PRIORITY = 300


@HOOKS.register_class()
class WandbCheckpointHook(Hook):
    """Wandb Checkpoint Hook for tracking model artifacts.
    Integrates with wandb to log model checkpoints as artifacts with detailed metadata.
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_CHECKPOINT_PRIORITY,
            'description': 'the priority for processing!'
        },
        'SAVE_CHECKPOINT': {
            'value': True,
            'description': 'Whether to save checkpoints as wandb artifacts!'
        },
        'CHECKPOINT_INTERVAL': {
            'value': 1000,
            'description': 'The interval for saving checkpoints as wandb artifacts!'
        },
        'SAVE_BEST': {
            'value': True,
            'description': 'Whether to save best checkpoints separately!'
        },
        'SAVE_BEST_BY': {
            'value': '',
            'description': 'Metric to track for determining best checkpoint!'
        },
        'TRACK_MODEL_FILES': {
            'value': True,
            'description': 'Whether to track the model files!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbCheckpointHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_CHECKPOINT_PRIORITY)
        self.save_checkpoint = cfg.get('SAVE_CHECKPOINT', True)
        self.checkpoint_interval = cfg.get('CHECKPOINT_INTERVAL', 1000)
        self.save_best = cfg.get('SAVE_BEST', True)
        self.save_best_by = cfg.get('SAVE_BEST_BY', '')
        self.track_model_files = cfg.get('TRACK_MODEL_FILES', True)
        
        # For tracking best metrics
        self.higher_the_best = True
        if self.save_best:
            if self.save_best_by.startswith('+'):
                self.save_best_by = self.save_best_by[1:]
            elif self.save_best_by.startswith('-'):
                self.save_best_by = self.save_best_by[1:]
                self.higher_the_best = False
        
        # Initialize best metric tracker
        self.best_metric = None
        self.best_step = None
        self.best_epoch = None
        
        # Keep track of the last artifact
        self.last_artifact = None
        self.wandb_run = None

    def before_solve(self, solver):
        """Get the wandb run if it exists."""
        if we.rank != 0:
            return
        
        try:
            # Get existing wandb run if available
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                solver.logger.warning("WandbCheckpointHook: No active wandb run found. "
                                     "Make sure to initialize wandb with WandbLogHook before this hook.")
            else:
                solver.logger.info(f"WandbCheckpointHook: Connected to wandb run: {self.wandb_run.name}")
                
                # Log initial model architecture information
                if hasattr(solver, 'model') and self.track_model_files:
                    try:
                        # Log model config as an artifact
                        model_config = self._get_model_config(solver)
                        if model_config:
                            model_config_artifact = wandb.Artifact(
                                name=f"model-config-{self.wandb_run.id}",
                                type="model-config",
                                description="Model architecture configuration"
                            )
                            with model_config_artifact.new_file("model_config.json") as f:
                                json.dump(model_config, f, indent=2)
                            self.wandb_run.log_artifact(model_config_artifact)
                    except Exception as e:
                        solver.logger.warning(f"Failed to log model config: {e}")
        except Exception as e:
            solver.logger.warning(f"Error in WandbCheckpointHook.before_solve: {e}")

    def after_iter(self, solver):
        """Log checkpoint artifacts at specified intervals."""
        if we.rank != 0 or self.wandb_run is None or not self.save_checkpoint:
            return
            
        try:
            # Save checkpoint at regular intervals
            if solver.total_iter > 0 and (
                    solver.total_iter % self.checkpoint_interval == 0 or 
                    solver.total_iter == solver.max_steps - 1):
                
                # Get checkpoint path that other hooks might have saved
                checkpoint_path = osp.join(
                    solver.work_dir,
                    f'checkpoints/ldm_step-{solver.total_iter + 1}.pth')
                
                if FS.exists(checkpoint_path):
                    # Create artifact for the checkpoint
                    artifact_name = f"model-checkpoint-step-{solver.total_iter + 1}"
                    checkpoint_artifact = wandb.Artifact(
                        name=artifact_name,
                        type="model-checkpoint",
                        description=f"Model checkpoint at step {solver.total_iter + 1}"
                    )
                    
                    # Add metadata to the artifact
                    checkpoint_artifact.metadata = {
                        "step": solver.total_iter + 1,
                        "epoch": solver.epoch,
                        "timestamp": time.time(),
                        "is_final": solver.total_iter == solver.max_steps - 1
                    }
                    
                    # Add performance metrics if available
                    if hasattr(solver, 'iter_outputs'):
                        for key, value in solver.iter_outputs.items():
                            if isinstance(value, (int, float)):
                                checkpoint_artifact.metadata[f"metric_{key}"] = value
                    
                    # Add the checkpoint file to the artifact
                    with FS.get_from(checkpoint_path, wait_finish=True) as local_file:
                        checkpoint_artifact.add_file(local_file, name=osp.basename(checkpoint_path))
                        
                    # Log the artifact to wandb
                    self.wandb_run.log_artifact(checkpoint_artifact)
                    self.last_artifact = checkpoint_artifact
                    
                    solver.logger.info(f"Logged checkpoint artifact to wandb: {artifact_name}")
                    
                    # Check for additional model files to track
                    if self.track_model_files:
                        model_dir = osp.join(
                            solver.work_dir,
                            f'checkpoints/ldm_step-{solver.total_iter + 1}')
                        
                        if FS.exists(model_dir):
                            # Create artifact for model files
                            model_artifact = wandb.Artifact(
                                name=f"model-files-step-{solver.total_iter + 1}",
                                type="model-files",
                                description=f"Model files at step {solver.total_iter + 1}"
                            )
                            
                            # Add model directory to artifact
                            with FS.get_dir_to_local_dir(model_dir) as local_dir:
                                model_artifact.add_dir(local_dir, name="model")
                                
                            # Log the artifact to wandb
                            self.wandb_run.log_artifact(model_artifact)
                            solver.logger.info(f"Logged model files artifact to wandb")
        except Exception as e:
            solver.logger.warning(f"Error in WandbCheckpointHook.after_iter: {e}")

    def after_epoch(self, solver):
        """Log epoch checkpoints and track best model."""
        if we.rank != 0 or self.wandb_run is None or not self.save_checkpoint:
            return
            
        try:
            # Create artifact for epoch checkpoint
            checkpoint_path = osp.join(solver.work_dir, f'epoch-{solver.epoch:05d}.pth')
            
            if FS.exists(checkpoint_path):
                # Track this epoch's checkpoint
                artifact_name = f"model-checkpoint-epoch-{solver.epoch}"
                checkpoint_artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model-checkpoint",
                    description=f"Model checkpoint at epoch {solver.epoch}"
                )
                
                # Add metadata
                checkpoint_artifact.metadata = {
                    "epoch": solver.epoch,
                    "step": solver.total_iter,
                    "timestamp": time.time(),
                    "is_final": solver.epoch == solver.max_epochs - 1
                }
                
                # Check if this is the best model by metric
                if self.save_best and self.save_best_by:
                    current_metric = None
                    
                    # Try to get current metric from epoch_outputs["eval"]
                    if 'eval' in solver.epoch_outputs and self.save_best_by in solver.epoch_outputs['eval']:
                        current_metric = solver.epoch_outputs['eval'][self.save_best_by]
                    
                    # Try from agg_iter_outputs if not found
                    elif 'eval' in solver.agg_iter_outputs and self.save_best_by in solver.agg_iter_outputs['eval']:
                        current_metric = solver.agg_iter_outputs['eval'][self.save_best_by]
                    
                    # Try from training metrics as fallback
                    elif 'train' in solver.agg_iter_outputs and self.save_best_by in solver.agg_iter_outputs['train']:
                        current_metric = solver.agg_iter_outputs['train'][self.save_best_by]
                    
                    # If metric found, check if it's the best
                    if current_metric is not None:
                        is_best = False
                        
                        if self.best_metric is None:
                            is_best = True
                        elif self.higher_the_best and current_metric > self.best_metric:
                            is_best = True
                        elif not self.higher_the_best and current_metric < self.best_metric:
                            is_best = True
                            
                        if is_best:
                            self.best_metric = current_metric
                            self.best_epoch = solver.epoch
                            self.best_step = solver.total_iter
                            
                            # Add best model indicator to metadata
                            checkpoint_artifact.metadata["is_best"] = True
                            checkpoint_artifact.metadata["best_metric_name"] = self.save_best_by
                            checkpoint_artifact.metadata["best_metric_value"] = current_metric
                            
                            # Log best model metrics to wandb
                            self.wandb_run.summary[f"best_{self.save_best_by}"] = current_metric
                            self.wandb_run.summary["best_epoch"] = solver.epoch
                            self.wandb_run.summary["best_step"] = solver.total_iter
                
                # Add performance metrics
                for mode, metrics in solver.epoch_outputs.items():
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            checkpoint_artifact.metadata[f"{mode}_{key}"] = value
                
                # Add checkpoint file to artifact
                with FS.get_from(checkpoint_path, wait_finish=True) as local_file:
                    checkpoint_artifact.add_file(local_file, name=osp.basename(checkpoint_path))
                
                # Log artifact to wandb
                self.wandb_run.log_artifact(checkpoint_artifact)
                self.last_artifact = checkpoint_artifact
                
                solver.logger.info(f"Logged epoch checkpoint artifact to wandb: {artifact_name}")
                
                # Handle pretrained checkpoint if it exists
                pretrain_path = osp.join(solver.work_dir, f'epoch-{solver.epoch:05d}_pretrain.pth')
                if FS.exists(pretrain_path):
                    pretrain_artifact = wandb.Artifact(
                        name=f"pretrain-model-epoch-{solver.epoch}",
                        type="pretrain-model",
                        description=f"Pretrained model weights at epoch {solver.epoch}"
                    )
                    
                    with FS.get_from(pretrain_path, wait_finish=True) as local_file:
                        pretrain_artifact.add_file(local_file, name=osp.basename(pretrain_path))
                    
                    self.wandb_run.log_artifact(pretrain_artifact)
                    solver.logger.info(f"Logged pretrained model artifact to wandb")
        except Exception as e:
            solver.logger.warning(f"Error in WandbCheckpointHook.after_epoch: {e}")

    def after_solve(self, solver):
        """Final cleanup and logging after training is complete."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Log final summary metrics
            self.wandb_run.summary["total_epochs"] = solver.epoch
            self.wandb_run.summary["total_steps"] = solver.total_iter
            
            # Log best model info if tracked
            if self.best_metric is not None:
                self.wandb_run.summary["best_metric"] = self.best_metric
                self.wandb_run.summary["best_epoch"] = self.best_epoch
                self.wandb_run.summary["best_step"] = self.best_step
            
            # Check for HuggingFace Hub push events
            if hasattr(solver, 'model') and hasattr(solver.model, 'push_to_hub'):
                self.wandb_run.summary["pushed_to_huggingface"] = True
            
            # Final sync
            self.wandb_run.log({}, commit=True)
        except Exception as e:
            solver.logger.warning(f"Error in WandbCheckpointHook.after_solve: {e}")

    def _get_model_config(self, solver):
        """Extract model configuration information."""
        model_config = {}
        
        if hasattr(solver, 'model'):
            if hasattr(solver.model, 'cfg'):
                model_config = solver.model.cfg
            
            # Add model architecture information
            try:
                model_type = type(solver.model).__name__
                model_config["model_type"] = model_type
                
                # Count parameters
                total_params = sum(p.numel() for p in solver.model.parameters())
                trainable_params = sum(p.numel() for p in solver.model.parameters() if p.requires_grad)
                
                model_config["total_parameters"] = total_params
                model_config["trainable_parameters"] = trainable_params
                model_config["frozen_parameters"] = total_params - trainable_params
            except Exception:
                pass
                
        return model_config

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbCheckpointHook.para_dict,
                            set_name=True)
