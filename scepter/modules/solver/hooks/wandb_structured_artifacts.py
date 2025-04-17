# -*- coding: utf-8 -*-
# Custom wandb hook for structured artifact saving

import os
import os.path as osp
import json
import time
import shutil
from typing import List, Dict, Any, Optional

import torch
import wandb

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

_DEFAULT_PRIORITY = 350


@HOOKS.register_class()
class WandbStructuredArtifactHook(Hook):
    """Custom hook for saving model artifacts in a structured way with specific directories."""
    
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_PRIORITY,
            'description': 'The priority for processing!'
        },
        'CHECKPOINT_INTERVAL': {
            'value': 100,
            'description': 'The interval for saving model artifacts!'
        },
        'SAVE_FINAL': {
            'value': True,
            'description': 'Whether to save final model artifacts!'
        },
        'ARTIFACT_MODEL_DIR': {
            'value': 'models',
            'description': 'Main directory for model components'
        },
        'MODEL_COMPONENTS': {
            'value': ['dit', 'text_encoder', 'tokenizer', 'vae'],
            'description': 'List of model component subdirectories'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbStructuredArtifactHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_PRIORITY)
        self.checkpoint_interval = cfg.get('CHECKPOINT_INTERVAL', 100)
        self.save_final = cfg.get('SAVE_FINAL', True)
        self.artifact_model_dir = cfg.get('ARTIFACT_MODEL_DIR', 'models')
        self.model_components = cfg.get('MODEL_COMPONENTS', ['dit', 'text_encoder', 'tokenizer', 'vae'])
        self.wandb_run = None
        self.last_artifact_step = -1
        
    def before_solve(self, solver):
        """Get the wandb run if it exists."""
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                solver.logger.warning("WandbStructuredArtifactHook: No active wandb run found. "
                                     "Make sure to initialize wandb with WandbLogHook before this hook.")
            else:
                solver.logger.info(f"WandbStructuredArtifactHook: Connected to wandb run: {self.wandb_run.name}")
                solver.logger.info(f"Will create structured artifacts with main dir '{self.artifact_model_dir}' "
                                  f"and components: {self.model_components}")
        except Exception as e:
            solver.logger.warning(f"Error in WandbStructuredArtifactHook.before_solve: {e}")
    
    def after_iter(self, solver):
        """Create and log structured model artifacts."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Only process at specified checkpoint intervals
        if solver.total_iter > 0 and (
                solver.total_iter % self.checkpoint_interval == 0 or 
                (self.save_final and solver.total_iter == solver.max_steps - 1)):
            
            # Skip if we've already processed this step
            if solver.total_iter == self.last_artifact_step:
                return
                
            self.last_artifact_step = solver.total_iter
            
            try:
                # Create a model artifact with proper structure
                artifact_name = f"ace-model-{solver.total_iter}"
                model_artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    description=f"ACE model checkpoint at step {solver.total_iter}"
                )
                
                # Add metadata to the artifact
                model_artifact.metadata = {
                    "step": solver.total_iter,
                    "epoch": solver.epoch,
                    "timestamp": time.time(),
                    "is_final": solver.total_iter == solver.max_steps - 1
                }
                
                # Add performance metrics if available
                if hasattr(solver, 'iter_outputs'):
                    for key, value in solver.iter_outputs.items():
                        if isinstance(value, (int, float)):
                            model_artifact.metadata[f"metric_{key}"] = value
                
                # Create a temporary directory with our desired structure
                import tempfile
                tmp_root = tempfile.mkdtemp()
                model_dir = os.path.join(tmp_root, self.artifact_model_dir)
                os.makedirs(model_dir, exist_ok=True)
                
                # Create subdirectories for each component
                for component in self.model_components:
                    os.makedirs(os.path.join(model_dir, component), exist_ok=True)
                
                # Find and organize model files
                checkpoint_dir = osp.join(solver.work_dir, f'checkpoints/ldm_step-{solver.total_iter}')
                checkpoint_file = osp.join(solver.work_dir, f'checkpoints/ldm_step-{solver.total_iter}.pth')
                
                # First check if there's a structured checkpoint directory
                if FS.exists(checkpoint_dir):
                    with FS.get_dir_to_local_dir(checkpoint_dir) as local_dir:
                        # Get all files recursively and organize them into our structure
                        for root, dirs, files in os.walk(local_dir):
                            for file in files:
                                src_path = os.path.join(root, file)
                                
                                # Determine which component this file belongs to
                                rel_path = os.path.relpath(src_path, local_dir)
                                component_match = None
                                
                                for component in self.model_components:
                                    if component in rel_path.lower():
                                        component_match = component
                                        break
                                
                                # Default to 'dit' if no match found
                                if component_match is None:
                                    component_match = 'dit'
                                
                                # Copy to the structured directory
                                dest_dir = os.path.join(model_dir, component_match)
                                os.makedirs(dest_dir, exist_ok=True)
                                dest_path = os.path.join(dest_dir, os.path.basename(src_path))
                                shutil.copy2(src_path, dest_path)
                
                # Add checkpoint file if it exists
                if FS.exists(checkpoint_file):
                    with FS.get_from(checkpoint_file, wait_finish=True) as local_file:
                        # Copy to the dit directory
                        dest_path = os.path.join(model_dir, 'dit', os.path.basename(local_file))
                        shutil.copy2(local_file, dest_path)
                
                # Also look for standard config files
                config_file = osp.join(solver.work_dir, f'checkpoints/configuration.json')
                if FS.exists(config_file):
                    with FS.get_from(config_file, wait_finish=True) as local_file:
                        dest_path = os.path.join(model_dir, 'dit', 'config.json')
                        shutil.copy2(local_file, dest_path)
                
                # Add the entire structured directory to the artifact
                model_artifact.add_dir(tmp_root, name="")
                
                # Log the artifact to wandb
                self.wandb_run.log_artifact(model_artifact)
                solver.logger.info(f"Logged structured model artifact to wandb: {artifact_name}")
                
                # Clean up temporary directory
                shutil.rmtree(tmp_root)
                
            except Exception as e:
                solver.logger.warning(f"Error in WandbStructuredArtifactHook.after_iter: {e}")
                import traceback
                solver.logger.warning(traceback.format_exc())
    
    def after_solve(self, solver):
        """Create a final artifact when training completes."""
        if we.rank != 0 or self.wandb_run is None or not self.save_final:
            return
            
        try:
            # Force creation of a final artifact
            if solver.total_iter != self.last_artifact_step:
                self.after_iter(solver)
                
            # Create a summary artifact with metadata
            summary_artifact = wandb.Artifact(
                name=f"ace-model-final-summary",
                type="model-summary",
                description=f"Final ACE model training summary"
            )
            
            # Add final metrics
            summary = {
                "total_steps": solver.total_iter,
                "total_epochs": solver.epoch,
                "completed_at": time.time(),
                "model_structure": {
                    "main_dir": self.artifact_model_dir,
                    "components": self.model_components
                }
            }
            
            # Add final metrics
            if hasattr(solver, 'agg_iter_outputs'):
                for phase in solver.agg_iter_outputs:
                    for key, value in solver.agg_iter_outputs[phase].items():
                        if isinstance(value, (int, float)):
                            summary[f"final_{phase}_{key}"] = value
            
            # Add the summary to the artifact
            with summary_artifact.new_file("summary.json") as f:
                json.dump(summary, f, indent=2)
                
            self.wandb_run.log_artifact(summary_artifact)
            solver.logger.info(f"Logged final model summary artifact")
            
        except Exception as e:
            solver.logger.warning(f"Error in WandbStructuredArtifactHook.after_solve: {e}")
    
    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                          __name__,
                          WandbStructuredArtifactHook.para_dict,
                          set_name=True)
