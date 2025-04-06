# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we

try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed, ParameterTrackerHook will not log to wandb")

_DEFAULT_TRACKER_PRIORITY = 90  # Lower than log hooks (100) to ensure it runs before them


@HOOKS.register_class()
class ParameterTrackerHook(Hook):
    """
    Hook to track model parameters, gradients, and other metrics at a granular level.
    This hook can work with or without wandb, storing detailed parameter statistics
    that can be used for debugging and analysis.
    """
    
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_TRACKER_PRIORITY,
            'description': 'The priority for processing!'
        },
        'TRACK_INTERVAL': {
            'value': 1,
            'description': 'Interval for tracking parameters (every N iterations)'
        },
        'TRACK_GRADIENTS': {
            'value': True,
            'description': 'Whether to track parameter gradients'
        },
        'TRACK_WEIGHTS': {
            'value': True,
            'description': 'Whether to track parameter weights'
        },
        'TRACK_OPTIMIZER_STATES': {
            'value': True,
            'description': 'Whether to track optimizer states (momentum, etc.)'
        },
        'DETAILED_TRACKING_INTERVAL': {
            'value': 100,
            'description': 'Interval for more detailed tracking (histograms, etc.)'
        },
        'TRACK_SPECIFIC_LAYERS': {
            'value': [],
            'description': 'List of specific layer names to track (empty = all)'
        },
        'TRACK_BATCH_INPUTS': {
            'value': False,
            'description': 'Whether to track statistics of batch inputs'
        },
        'TRACK_BATCH_OUTPUTS': {
            'value': False,
            'description': 'Whether to track statistics of batch outputs'
        },
        'LOG_TO_WANDB': {
            'value': True,
            'description': 'Whether to log tracked parameters to wandb'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(ParameterTrackerHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_TRACKER_PRIORITY)
        self.track_interval = cfg.get('TRACK_INTERVAL', 1)
        self.track_gradients = cfg.get('TRACK_GRADIENTS', True)
        self.track_weights = cfg.get('TRACK_WEIGHTS', True)
        self.track_optimizer_states = cfg.get('TRACK_OPTIMIZER_STATES', True)
        self.detailed_tracking_interval = cfg.get('DETAILED_TRACKING_INTERVAL', 100)
        self.track_specific_layers = cfg.get('TRACK_SPECIFIC_LAYERS', [])
        self.track_batch_inputs = cfg.get('TRACK_BATCH_INPUTS', False)
        self.track_batch_outputs = cfg.get('TRACK_BATCH_OUTPUTS', False)
        self.log_to_wandb = cfg.get('LOG_TO_WANDB', True)
        
        # Initialize storage for tracked parameters
        self.parameter_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.optimizer_state_history = defaultdict(list)
        self.batch_stats_history = defaultdict(list)
        
        # Hooks for tracking intermediate activations
        self.activation_hooks = []
        self.current_activations = {}
        
        # Flag to check if wandb is available
        self.wandb_available = False
        try:
            import wandb
            self.wandb_available = 'run' in dir(wandb) and wandb.run is not None
        except ImportError:
            self.wandb_available = False
            
    def before_solve(self, solver):
        """Set up hooks for tracking activations."""
        if we.rank != 0:
            return
            
        # Register hooks for tracking activations if requested
        if self.track_specific_layers and hasattr(solver, 'model') and solver.model is not None:
            self._register_activation_hooks(solver.model)
    
    def _register_activation_hooks(self, model):
        """Register hooks to capture activations from specific layers."""
        # Clear any existing hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        # Function to capture activations
        def hook_fn(name):
            def fn(module, input, output):
                # Store the activation
                if isinstance(output, torch.Tensor):
                    self.current_activations[name] = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    self.current_activations[name] = output[0].detach()
            return fn
        
        # Register hooks for specified layers
        for name, module in model.named_modules():
            if not self.track_specific_layers or any(layer_name in name for layer_name in self.track_specific_layers):
                self.activation_hooks.append(module.register_forward_hook(hook_fn(name)))
                
        if self.logger:
            self.logger.info(f"Registered activation hooks for {len(self.activation_hooks)} layers")
    
    def _track_parameter_stats(self, name, param):
        """Track statistics for a parameter."""
        if not param.numel():
            return {}
            
        # Basic statistics that are cheap to compute
        stats = {
            'mean': param.mean().item(),
            'std': param.std().item() if param.numel() > 1 else 0,
            'min': param.min().item(),
            'max': param.max().item(),
            'norm': torch.norm(param).item(),
        }
        
        # Add sparsity (percentage of zeros)
        zeros = (param == 0).float().mean().item() * 100
        stats['sparsity_pct'] = zeros
        
        return stats
    
    def _track_batch_stats(self, data, prefix):
        """Track statistics for batch inputs or outputs."""
        stats = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    key_stats = self._track_parameter_stats(f"{prefix}/{key}", value)
                    for stat_name, stat_value in key_stats.items():
                        stats[f"{prefix}/{key}/{stat_name}"] = stat_value
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    key_stats = self._track_parameter_stats(f"{prefix}/{i}", item)
                    for stat_name, stat_value in key_stats.items():
                        stats[f"{prefix}/{i}/{stat_name}"] = stat_value
        elif isinstance(data, torch.Tensor):
            key_stats = self._track_parameter_stats(prefix, data)
            for stat_name, stat_value in key_stats.items():
                stats[f"{prefix}/{stat_name}"] = stat_value
                
        return stats
    
    def before_iter(self, solver):
        """Track batch inputs if requested."""
        if we.rank != 0 or not self.track_batch_inputs:
            return
            
        if solver.total_iter % self.track_interval == 0 and hasattr(solver, 'batch_data'):
            try:
                batch_stats = self._track_batch_stats(solver.batch_data, 'batch_inputs')
                self.batch_stats_history['inputs'].append(batch_stats)
                
                # Log to wandb if available
                if self.log_to_wandb and self.wandb_available:
                    wandb.log(batch_stats, step=solver.total_iter)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error tracking batch inputs: {e}")
    
    def after_iter(self, solver):
        """Track parameters, gradients, and batch outputs after each iteration."""
        if we.rank != 0:
            return
            
        # Only track at specified intervals
        if solver.total_iter % self.track_interval != 0:
            return
            
        try:
            metrics = {}
            
            # Track model parameters and gradients
            if hasattr(solver, 'model') and solver.model is not None:
                model = solver.model
                
                for name, param in model.named_parameters():
                    # Skip if we're only tracking specific layers and this isn't one of them
                    if self.track_specific_layers and not any(layer_name in name for layer_name in self.track_specific_layers):
                        continue
                        
                    # Track weights
                    if self.track_weights:
                        param_stats = self._track_parameter_stats(param)
                        for stat_name, stat_value in param_stats.items():
                            metrics[f"parameters/{name}/{stat_name}"] = stat_value
                            
                    # Track gradients
                    if self.track_gradients and param.grad is not None:
                        grad_stats = self._track_parameter_stats(param.grad)
                        for stat_name, stat_value in grad_stats.items():
                            metrics[f"gradients/{name}/{stat_name}"] = stat_value
                            
                    # Add detailed tracking at less frequent intervals
                    if solver.total_iter % self.detailed_tracking_interval == 0:
                        # Add histograms if wandb is available
                        if self.log_to_wandb and self.wandb_available:
                            if self.track_weights:
                                metrics[f"histograms/parameters/{name}"] = wandb.Histogram(
                                    param.detach().cpu().numpy()
                                )
                            if self.track_gradients and param.grad is not None:
                                metrics[f"histograms/gradients/{name}"] = wandb.Histogram(
                                    param.grad.detach().cpu().numpy()
                                )
            
            # Track optimizer states
            if self.track_optimizer_states and hasattr(solver, 'optimizer') and solver.optimizer is not None:
                for i, param_group in enumerate(solver.optimizer.param_groups):
                    # Log learning rate
                    if 'lr' in param_group:
                        metrics[f"optimizer/group_{i}/lr"] = param_group['lr']
                        
                    # Log other hyperparameters in the param group
                    for key, value in param_group.items():
                        if key != 'params' and key != 'lr' and isinstance(value, (int, float)):
                            metrics[f"optimizer/group_{i}/{key}"] = value
                            
                    # Track momentum buffers if available (for SGD with momentum)
                    if hasattr(solver.optimizer, 'state'):
                        for j, param in enumerate(param_group['params']):
                            if param in solver.optimizer.state:
                                state = solver.optimizer.state[param]
                                for state_key, state_value in state.items():
                                    if isinstance(state_value, torch.Tensor):
                                        state_stats = self._track_parameter_stats(state_value)
                                        for stat_name, stat_value in state_stats.items():
                                            metrics[f"optimizer/state/group_{i}/param_{j}/{state_key}/{stat_name}"] = stat_value
            
            # Track batch outputs
            if self.track_batch_outputs and hasattr(solver, 'iter_outputs'):
                batch_output_stats = self._track_batch_stats(solver.iter_outputs, 'batch_outputs')
                for key, value in batch_output_stats.items():
                    metrics[key] = value
                    
            # Track activations from hooks
            for name, activation in self.current_activations.items():
                act_stats = self._track_parameter_stats(activation)
                for stat_name, stat_value in act_stats.items():
                    metrics[f"activations/{name}/{stat_name}"] = stat_value
                    
                # Add histograms at less frequent intervals
                if solver.total_iter % self.detailed_tracking_interval == 0 and self.log_to_wandb and self.wandb_available:
                    metrics[f"histograms/activations/{name}"] = wandb.Histogram(
                        activation.detach().cpu().reshape(-1).numpy()
                    )
            
            # Log all metrics to wandb
            if self.log_to_wandb and self.wandb_available:
                wandb.log(metrics, step=solver.total_iter)
                
            # Store metrics in history
            for key, value in metrics.items():
                if 'parameters/' in key:
                    self.parameter_history[key].append((solver.total_iter, value))
                elif 'gradients/' in key:
                    self.gradient_history[key].append((solver.total_iter, value))
                elif 'optimizer/' in key:
                    self.optimizer_state_history[key].append((solver.total_iter, value))
                    
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in ParameterTrackerHook.after_iter: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
    
    def after_solve(self, solver):
        """Clean up hooks and resources."""
        # Remove activation hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
    @staticmethod
    def get_config_template():
        return dict_to_yaml('PARAMETER_TRACKER_HOOK', {
            'PRIORITY': _DEFAULT_TRACKER_PRIORITY,
            'TRACK_INTERVAL': 1,
            'TRACK_GRADIENTS': True,
            'TRACK_WEIGHTS': True,
            'TRACK_OPTIMIZER_STATES': True,
            'DETAILED_TRACKING_INTERVAL': 100,
            'TRACK_SPECIFIC_LAYERS': [],
            'TRACK_BATCH_INPUTS': False,
            'TRACK_BATCH_OUTPUTS': False,
            'LOG_TO_WANDB': True
        })
