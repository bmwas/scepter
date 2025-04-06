# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import warnings
from typing import Dict, Any, Optional, List

import torch
from scepter.modules.solver.hooks.lr import LrHook, _get_lr_from_scheduler
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we

try:
    import wandb
except Exception as e:
    warnings.warn(f'Running without wandb! {e}')

_DEFAULT_LR_PRIORITY = 200


@HOOKS.register_class()
class WandbLrHook(LrHook):
    """ Enhanced Learning rate updater hook with Weights & Biases integration.
    
    Extends the LrHook functionality to track learning rates in wandb.
    Logs learning rate changes, schedule visualization, and per-group learning rates.
    
    Args:
        set_by_epoch (bool): Reset learning rate by epoch, we recommend true if solver.num_folds == 1
        warmup_func (str, None): Do not warm up if None, currently support linear warmup
        warmup_epochs (int): Number of warmup epochs
        warmup_start_lr (float): Starting learning rate for warmup
        log_lr_schedule (bool): Whether to log the entire learning rate schedule to wandb
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LR_PRIORITY,
            'description': 'the priority for processing!'
        },
        'WARMUP_FUNC': {
            'value': 'linear',
            'description': 'Only linear warmup supported!'
        },
        'WARMUP_EPOCHS': {
            'value': 1,
            'description': 'The warmup epochs!'
        },
        'WARMUP_START_LR': {
            'value': 0.0001,
            'description': 'The warmup start learning rate!'
        },
        'SET_BY_EPOCH': {
            'value': True,
            'description': 'Set the learning rate by epoch!'
        },
        'LOG_LR_SCHEDULE': {
            'value': True,
            'description': 'Log the entire learning rate schedule as a chart!'
        },
        'SCHEDULE_LOGGING_STEPS': {
            'value': 100,
            'description': 'Number of points to log in the LR schedule chart!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbLrHook, self).__init__(cfg, logger=logger)
        self.log_lr_schedule = cfg.get('LOG_LR_SCHEDULE', True)
        self.schedule_logging_steps = cfg.get('SCHEDULE_LOGGING_STEPS', 100)
        
        # Wandb run reference
        self.wandb_run = None
        
        # Track parameter groups for detailed logging
        self.param_group_names = []
        
        # Track LR changes for logging
        self.last_logged_lrs = {}

    def before_solve(self, solver):
        """Connect to wandb and log initial learning rate schedule."""
        # Call parent method first to set up warmup
        super().before_solve(solver)
        
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run if available
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                solver.logger.warning("WandbLrHook: No active wandb run found. "
                                    "Make sure to initialize wandb with WandbLogHook before this hook.")
            else:
                solver.logger.info(f"WandbLrHook: Connected to wandb run: {self.wandb_run.name}")
                
                # Initialize parameter group names for detailed logging
                if hasattr(solver, 'optimizer'):
                    for i, param_group in enumerate(solver.optimizer.param_groups):
                        name = param_group.get('name', f'group_{i}')
                        self.param_group_names.append(name)
                        # Log initial learning rate
                        self.last_logged_lrs[name] = param_group['lr']
                
                # Log learning rate schedule if requested
                if self.log_lr_schedule and hasattr(solver, 'lr_scheduler') and solver.lr_scheduler is not None:
                    self._log_lr_schedule(solver)
        except Exception as e:
            solver.logger.warning(f"Error in WandbLrHook.before_solve: {e}")

    def after_epoch(self, solver):
        """Log learning rate changes after each epoch."""
        # Call parent method first to update learning rates
        super().after_epoch(solver)
        
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Log current learning rates for each parameter group
            if solver.is_train_mode and hasattr(solver, 'optimizer'):
                lr_dict = {}
                
                for i, param_group in enumerate(solver.optimizer.param_groups):
                    name = self.param_group_names[i] if i < len(self.param_group_names) else f'group_{i}'
                    current_lr = param_group['lr']
                    
                    # Log to wandb
                    lr_dict[f'learning_rate/{name}'] = current_lr
                    
                    # Log change from previous value
                    if name in self.last_logged_lrs:
                        prev_lr = self.last_logged_lrs[name]
                        if prev_lr != current_lr:
                            # Calculate change in percentage
                            change_pct = (current_lr - prev_lr) / prev_lr * 100 if prev_lr != 0 else float('inf')
                            lr_dict[f'learning_rate_change/{name}'] = change_pct
                            
                    # Update last logged value
                    self.last_logged_lrs[name] = current_lr
                
                # Add epoch information
                lr_dict['epoch'] = solver.epoch
                
                # Log to wandb
                if lr_dict:
                    self.wandb_run.log(lr_dict, step=solver.total_iter)
        except Exception as e:
            solver.logger.warning(f"Error in WandbLrHook.after_epoch: {e}")

    def before_iter(self, solver):
        """Log learning rate changes during training iterations."""
        # Call parent method first to update learning rates if not epoch-based
        super().before_iter(solver)
        
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Only log periodically during training to avoid excessive logging
            if (not self.set_by_epoch 
                and solver.is_train_mode 
                and hasattr(solver, 'optimizer')
                and solver.iter % 100 == 0):  # Log every 100 iterations
                
                lr_dict = {}
                
                for i, param_group in enumerate(solver.optimizer.param_groups):
                    name = self.param_group_names[i] if i < len(self.param_group_names) else f'group_{i}'
                    current_lr = param_group['lr']
                    
                    # Log to wandb
                    lr_dict[f'learning_rate/{name}'] = current_lr
                
                # Log to wandb
                if lr_dict:
                    self.wandb_run.log(lr_dict, step=solver.total_iter)
        except Exception as e:
            solver.logger.warning(f"Error in WandbLrHook.before_iter: {e}")

    def _log_lr_schedule(self, solver):
        """Log the entire learning rate schedule as a chart in wandb."""
        if not hasattr(solver, 'lr_scheduler') or solver.lr_scheduler is None:
            return
            
        try:
            # Create data for the learning rate schedule chart
            schedule_data = []
            
            # Determine total epochs/steps
            total_epochs = solver.max_epochs
            
            # Generate points for the chart
            step_size = max(1, total_epochs // self.schedule_logging_steps)
            
            # Include warmup period if applicable
            if self.warmup_func is not None and self.warmup_epochs > 0:
                # Add more granularity during warmup
                warmup_steps = min(20, self.warmup_epochs * 5)
                for i in range(warmup_steps + 1):
                    epoch = i * self.warmup_epochs / warmup_steps
                    if epoch <= self.warmup_epochs:
                        lr = self._get_warmup_lr(epoch)
                        schedule_data.append([epoch, lr])
            
            # Add points for the rest of training
            for epoch in range(max(0, self.warmup_epochs), total_epochs, step_size):
                lr = _get_lr_from_scheduler(solver.lr_scheduler, epoch)
                schedule_data.append([epoch, lr])
            
            # Add the final epoch
            if total_epochs - 1 not in [x[0] for x in schedule_data]:
                lr = _get_lr_from_scheduler(solver.lr_scheduler, total_epochs - 1)
                schedule_data.append([total_epochs - 1, lr])
            
            # Create wandb table for the chart
            columns = ["epoch", "learning_rate"]
            lr_table = wandb.Table(columns=columns, data=schedule_data)
            
            # Log to wandb
            self.wandb_run.log({
                "learning_rate_schedule": wandb.plot.line(
                    lr_table, "epoch", "learning_rate",
                    title="Learning Rate Schedule"
                )
            })
            
            # Create a summary of the learning rate schedule
            schedule_summary = {
                "initial_lr": schedule_data[0][1] if schedule_data else None,
                "final_lr": schedule_data[-1][1] if schedule_data else None,
                "warmup_epochs": self.warmup_epochs if self.warmup_func is not None else 0,
                "warmup_start_lr": self.warmup_start_lr if self.warmup_func is not None else None,
                "scheduler_type": type(solver.lr_scheduler).__name__
            }
            
            # Add to wandb summary
            for key, value in schedule_summary.items():
                if value is not None:
                    self.wandb_run.summary[f"lr_schedule_{key}"] = value
                    
        except Exception as e:
            solver.logger.warning(f"Error logging LR schedule: {e}")

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbLrHook.para_dict,
                            set_name=True)
