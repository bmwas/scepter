# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
import wandb
import torch
import numpy as np
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.distribute import we

@HOOKS.register_class()
class CustomWandbLossHook(Hook):
    """A simplified hook that focuses only on logging loss values to wandb.
    
    This hook extracts loss values from solver outputs and logs them directly to wandb
    as scalar values, with no additional processing or transformation.
    """
    
    para_dict = [{
        'PRIORITY': {
            'value': 150,  # Higher than standard log hooks
            'description': 'Priority for hook execution order'
        },
        'LOG_INTERVAL': {
            'value': 10,
            'description': 'Interval for logging to wandb'
        },
        'METRICS': {
            'value': ['loss'],
            'description': 'Metrics to extract and log'
        },
        'PREFIX': {
            'value': 'metrics/',
            'description': 'Prefix for metric names in wandb'
        }
    }]
    
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', 150)
        self.log_interval = cfg.get('LOG_INTERVAL', 10)
        self.metrics = cfg.get('METRICS', ['loss'])
        self.prefix = cfg.get('PREFIX', 'metrics/')
        
        # Track last log time to avoid excessive logging
        self.last_log_time = 0
        
        # Initialize wandb connection
        self.wandb_run = None
        
    def before_solve(self, solver):
        """Connect to existing wandb run."""
        if we.rank != 0:
            return
            
        try:
            # Try to connect to existing wandb run
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                if self.logger:
                    self.logger.warning("CustomWandbLossHook: No active wandb run found.")
            else:
                if self.logger:
                    self.logger.info(f"CustomWandbLossHook: Connected to wandb run: {self.wandb_run.name}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in CustomWandbLossHook.before_solve: {e}")
    
    def after_iter(self, solver):
        """Extract loss values and log them to wandb."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Only log at specified intervals
        if solver.total_iter % self.log_interval != 0:
            return
            
        try:
            # Get outputs from solver
            outputs = {}
            
            # Try different attribute names that might contain loss
            if hasattr(solver, 'iter_outputs'):
                outputs.update(solver.iter_outputs)
                
            if hasattr(solver, 'epoch_outputs'):
                outputs.update(solver.epoch_outputs)
                
            if hasattr(solver, 'collect_log_vars'):
                outputs.update(solver.collect_log_vars())
                
            log_dict = {}
            
            # Extract metrics we care about
            for metric in self.metrics:
                if metric in outputs:
                    value = outputs[metric]
                    
                    # Convert tensor to number
                    if isinstance(value, torch.Tensor):
                        if value.dim() == 0:  # Scalar tensor
                            value = value.item()
                        else:  # Multi-dimensional tensor
                            value = value.mean().item()
                    
                    # Convert list/array to number
                    elif isinstance(value, (list, np.ndarray)):
                        if len(value) > 0:
                            # Convert all elements to float if possible
                            try:
                                value = [float(v) for v in value]
                                value = sum(value) / len(value)  # Average
                            except (TypeError, ValueError):
                                # If conversion fails, skip this metric
                                continue
                    
                    # Skip non-numeric values
                    if not isinstance(value, (int, float)):
                        continue
                        
                    # Add to log dict
                    log_key = f"{self.prefix}{solver.mode}/{metric}"
                    log_dict[log_key] = value
                    
                    # Also log raw value with different prefix
                    log_dict[f"raw/{solver.mode}/{metric}"] = value
            
            # Add step information
            log_dict["iteration"] = solver.total_iter
            log_dict["epoch"] = getattr(solver, "epoch", 0)
            
            # Log to wandb if we have metrics
            if log_dict:
                self.wandb_run.log(log_dict, step=solver.total_iter)
                
                if self.logger and hasattr(self, 'debug') and self.debug:
                    self.logger.info(f"CustomWandbLossHook: Logged metrics: {log_dict}")
                    
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in CustomWandbLossHook.after_iter: {e}")
    
    def after_epoch(self, solver):
        """Log epoch-level metrics."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Get epoch outputs
            if hasattr(solver, 'epoch_outputs'):
                outputs = solver.epoch_outputs.copy()
                
                log_dict = {}
                
                # Extract metrics we care about
                for metric in self.metrics:
                    if metric in outputs:
                        value = outputs[metric]
                        
                        # Convert tensor to number
                        if isinstance(value, torch.Tensor):
                            if value.dim() == 0:  # Scalar tensor
                                value = value.item()
                            else:  # Multi-dimensional tensor
                                value = value.mean().item()
                        
                        # Convert list/array to number
                        elif isinstance(value, (list, np.ndarray)):
                            if len(value) > 0:
                                # Convert all elements to float if possible
                                try:
                                    value = [float(v) for v in value]
                                    value = sum(value) / len(value)  # Average
                                except (TypeError, ValueError):
                                    # If conversion fails, skip this metric
                                    continue
                        
                        # Skip non-numeric values
                        if not isinstance(value, (int, float)):
                            continue
                            
                        # Add to log dict
                        log_key = f"{self.prefix}epoch/{metric}"
                        log_dict[log_key] = value
                
                # Log to wandb if we have metrics
                if log_dict:
                    self.wandb_run.log(log_dict, step=solver.total_iter)
                    
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in CustomWandbLossHook.after_epoch: {e}")
