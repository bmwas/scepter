# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import warnings
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from scepter.modules.solver.hooks.val_loss import ValLossHook, float_format
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

try:
    import wandb
    from PIL import Image
except Exception as e:
    warnings.warn(f'Running without wandb or PIL! {e}')

_DEFAULT_VAL_PRIORITY = 200


@HOOKS.register_class()
class WandbValLossHook(ValLossHook):
    """Enhanced Validation Loss Hook with Weights & Biases integration.
    
    Extends the ValLossHook functionality to track validation losses in wandb.
    Logs detailed validation metrics, creates interactive charts and artifacts.
    
    Args:
        All arguments from ValLossHook
        log_detailed_metrics (bool): Whether to log detailed metrics for each sample
        create_evaluation_table (bool): Whether to create a table with all validation samples
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_VAL_PRIORITY,
            'description': 'The priority for processing!'
        },
        'VAL_INTERVAL': {
            'value': 1000,
            'description': 'the interval for log print!'
        },
        'VAL_LIMITATION_SIZE': {
            'value': 1000000,
            'description': 'the limitation size for validation!'
        },
        'VAL_SEED': {
            'value': 2025,
            'description': 'the validation seed for t or generator sample!'
        },
        'LOG_DETAILED_METRICS': {
            'value': True,
            'description': 'Whether to log detailed metrics for each sample!'
        },
        'CREATE_EVALUATION_TABLE': {
            'value': True,
            'description': 'Whether to create a table with all validation samples!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbValLossHook, self).__init__(cfg, logger=logger)
        self.log_detailed_metrics = cfg.get('LOG_DETAILED_METRICS', True)
        self.create_evaluation_table = cfg.get('CREATE_EVALUATION_TABLE', True)
        
        # Wandb run reference
        self.wandb_run = None
        
        # Track evaluation tables
        self.eval_table = None

    def before_solve(self, solver):
        """Connect to existing wandb run if available."""
        # Call parent method first to initialize validation data
        super().before_solve(solver)
        
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run if available
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                solver.logger.warning("WandbValLossHook: No active wandb run found. "
                                    "Make sure to initialize wandb with WandbLogHook before this hook.")
            else:
                solver.logger.info(f"WandbValLossHook: Connected to wandb run: {self.wandb_run.name}")
                
                # Initialize evaluation table if needed
                if self.create_evaluation_table:
                    # Create evaluation table for validation results
                    self.eval_table = wandb.Table(
                        columns=["step", "sample_id"] + 
                                self.meta_field + 
                                ["loss", "global_avg_loss"]
                    )
        except Exception as e:
            solver.logger.warning(f"Error in WandbValLossHook.before_solve: {e}")

    def save_record(self, solver, compute_results, all_loss, step):
        """Extended save_record to also log validation results to wandb."""
        # First call the parent method to save to file system
        super().save_record(solver, compute_results, all_loss, step)
        
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            self.logger.info(f"WandbValLossHook: compute_results keys at step {step}: {list(compute_results.keys())}")
            # Log only the 'all' validation scalar to wandb
            if 'all' in compute_results:
                val_loss_all = float(compute_results['all'])
                log_dict = {"val_loss/all": val_loss_all}
                self.logger.info(f"Logging val_loss/all to wandb: {val_loss_all} at step {step}")
                self.wandb_run.log(log_dict, step=step)
                try:
                    self.wandb_run.flush()
                except Exception as e:
                    self.logger.warning(f"wandb flush failed: {e}")
            
            # Log detailed metrics if requested
            if self.log_detailed_metrics:
                # Process individual loss entries
                for loss_item in all_loss:
                    # Skip entries that don't have a loss value
                    if not isinstance(loss_item, dict) or 'loss' not in loss_item:
                        continue
                        
                    # Include loss value in evaluation table
                    if self.create_evaluation_table and self.eval_table is not None:
                        row_data = [step, loss_item.get('sample_id', 0)]
                        for field in self.meta_field:
                            row_data.append(loss_item.get(field, ''))
                        row_data.append(loss_item.get('loss', 0))
                        row_data.append(compute_results.get('all', 0))
                        self.eval_table.add_data(*row_data)
            
            # --- Log the best validation result to wandb ---
            # Find the minimum 'all' value and its step from results['summary']['all']
            save_folder = os.path.join(solver.work_dir, self.save_folder)
            summary_path = os.path.join(save_folder, 'curve', 'all.png')
            if FS.exists(summary_path):
                try:
                    with FS.get_from(summary_path, wait_finish=True) as local_path:
                        log_dict = {"val_loss/best_curve": wandb.Image(local_path)}
                        self.wandb_run.log(log_dict, step=step)
                except Exception as e:
                    solver.logger.warning(f"Error logging best validation curve: {e}")
            # Log best scalar value
            if 'all' in compute_results:
                # Load history for best validation loss
                try:
                    history_bytes = FS.get_object(os.path.join(solver.work_dir, self.save_folder, 'history.json'))
                    history = json.loads(history_bytes.decode())
                    all_summary = history.get('summary', {}).get('all', {})
                    if all_summary:
                        best_val_loss = min(all_summary.values())
                        best_step = min(all_summary, key=all_summary.get)
                    else:
                        best_val_loss = float(compute_results['all'])
                        best_step = step
                except Exception as e:
                    solver.logger.warning(f"Error loading history.json for best val loss: {e}")
                    best_val_loss = float(compute_results['all'])
                    best_step = step
                self.wandb_run.summary['best_val_loss'] = best_val_loss
                self.wandb_run.summary['best_val_loss_step'] = best_step
            
            # Save a copy of the history.json file as a wandb artifact
            save_folder = os.path.join(solver.work_dir, self.save_folder)
            save_history = os.path.join(save_folder, 'history.json')
            
            # Create an artifact to store the validation history
            if FS.exists(save_history):
                try:
                    # Download the file if it's remote
                    with FS.get_from(save_history, wait_finish=True) as local_path:
                        # Create and log the artifact
                        artifact = wandb.Artifact(
                            name=f"val_loss_history_{step}", 
                            type="validation_history"
                        )
                        artifact.add_file(local_path, "history.json")
                        self.wandb_run.log_artifact(artifact)
                except Exception as e:
                    solver.logger.warning(f"Error saving validation history artifact: {e}")
            
            # Log validation curves if they exist
            curve_folder = os.path.join(save_folder, 'curve')
            if FS.exists(curve_folder):
                summary_path = os.path.join(curve_folder, 'summary.png')
                if FS.exists(summary_path):
                    try:
                        with FS.get_from(summary_path, wait_finish=True) as local_path:
                            log_dict = {"val_loss/summary_curve": wandb.Image(local_path)}
                            self.wandb_run.log(log_dict, step=step)
                    except Exception as e:
                        solver.logger.warning(f"Error logging validation curve: {e}")
                
                # Log detailed curves if they exist
                detail_folder = os.path.join(curve_folder, 'detail')
                if FS.exists(detail_folder):
                    curve_images = {}
                    try:
                        # List all PNG files in the detail folder
                        for filename in os.listdir(detail_folder):
                            if filename.endswith('.png'):
                                file_path = os.path.join(detail_folder, filename)
                                curve_name = os.path.splitext(filename)[0]
                                curve_images[f"val_loss/detail/{curve_name}"] = wandb.Image(file_path)
                        if curve_images:
                            self.wandb_run.log(curve_images, step=step)
                    except Exception as e:
                        solver.logger.warning(f"Error logging detailed validation curves: {e}")
                        
        except Exception as e:
            solver.logger.warning(f"Error in WandbValLossHook.save_record: {e}")

    def after_solve(self, solver):
        """Log final evaluation table."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Log evaluation table if it exists and contains data
            if self.create_evaluation_table and self.eval_table is not None:
                if hasattr(self.eval_table, 'data') and len(self.eval_table.data) > 0:
                    self.wandb_run.log({"validation/evaluation_table": self.eval_table})
                    
            # Log any final validation artifacts
            save_folder = os.path.join(solver.work_dir, self.save_folder)
            save_history = os.path.join(save_folder, 'history.json')
            
            if FS.exists(save_history):
                try:
                    # Download the file if it's remote
                    with FS.get_from(save_history, wait_finish=True) as local_path:
                        # Create and log the artifact
                        artifact = wandb.Artifact(
                            name="final_val_loss_history", 
                            type="validation_history"
                        )
                        artifact.add_file(local_path, "history.json")
                        self.wandb_run.log_artifact(artifact)
                        
                        # Also add to summary
                        with open(local_path, 'r') as f:
                            history_data = json.load(f)
                            if 'summary' in history_data:
                                # Find the best validation loss
                                for metric, values in history_data['summary'].items():
                                    if values:
                                        best_step = min(values.items(), key=lambda x: float(x[1]))[0]
                                        best_value = values[best_step]
                                        self.wandb_run.summary[f"best_val_{metric}"] = best_value
                                        self.wandb_run.summary[f"best_val_{metric}_step"] = int(best_step)
                except Exception as e:
                    solver.logger.warning(f"Error saving final validation history artifact: {e}")
                    
        except Exception as e:
            solver.logger.warning(f"Error in WandbValLossHook.after_solve: {e}")

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbValLossHook.para_dict,
                            set_name=True)
