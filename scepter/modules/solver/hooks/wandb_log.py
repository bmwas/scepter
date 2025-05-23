# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import os
import os.path as osp
import time
import warnings
from collections import defaultdict
from typing import Optional, Dict, Any

import json
import numpy as np
import torch
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import LogAgg, time_since
import yaml

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
        },
        'TRACK_ACTIVATIONS': {
            'value': False,
            'description': 'whether to track activations in the model!'
        },
        'ACTIVATION_LAYERS': {
            'value': [],
            'description': 'specific layers to track activations for!'
        },
        'ACTIVATION_FREQUENCY': {
            'value': 100,
            'description': 'frequency of tracking activations!'
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
        
        # Track activations
        self.track_activations = cfg.get('TRACK_ACTIVATIONS', False)
        self.activation_layers = cfg.get('ACTIVATION_LAYERS', [])
        self.activation_frequency = cfg.get('ACTIVATION_FREQUENCY', 100)
        self.activation_hooks = []
        self.activations = {}
        
        # Always log loss at every iteration regardless of interval
        self.log_loss_every_iter = True
        
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
            if self.run_name:
                try:
                    self.wandb_run.name = self.run_name
                except AttributeError:
                    if solver and hasattr(solver, 'logger'):
                        solver.logger.warning(f"Cannot modify name of existing wandb run")
                    else:
                        print(f"Cannot modify name of existing wandb run")
            if self.tags:
                try:
                    self.wandb_run.tags = self.wandb_run.tags + tuple(self.tags)
                except AttributeError:
                    if solver and hasattr(solver, 'logger'):
                        solver.logger.warning(f"Cannot modify tags of existing wandb run")
                    else:
                        print(f"Cannot modify tags of existing wandb run")
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
        """Initialize wandb run and set up logging."""
        if we.rank != 0:
            return

        try:
            # Set up wandb logging directory
            if self.log_dir is None:
                self.log_dir = os.path.join(solver.work_dir, 'wandb')
            self._local_log_dir, _ = FS.map_to_local(self.log_dir)
            os.makedirs(self._local_log_dir, exist_ok=True)
            
            # Initialize wandb if not already initialized
            if self.wandb_run is None:
                self._init_wandb(solver)
                
            # Register activation hooks if requested
            if self.track_activations and hasattr(solver, 'model') and solver.model is not None:
                self._register_activation_hooks(solver.model)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbLogHook.before_solve: {e}")

    def _register_activation_hooks(self, model):
        """Register hooks to track activations in the model."""
        if not self.track_activations:
            return
            
        try:
            # Clear any existing hooks
            for hook in self.activation_hooks:
                hook.remove()
            self.activation_hooks = []
            
            # Function to capture activations
            def hook_fn(name):
                def fn(module, input, output):
                    # Store the activation
                    if isinstance(output, torch.Tensor):
                        self.activations[name] = output.detach()
                    elif isinstance(output, tuple) and len(output) > 0:
                        self.activations[name] = output[0].detach()
                return fn
            
            # Register hooks for all modules or specific ones
            if not self.activation_layers:
                # Track all modules with parameters
                for name, module in model.named_modules():
                    if list(module.parameters()):  # Only modules with parameters
                        self.activation_hooks.append(module.register_forward_hook(hook_fn(name)))
            else:
                # Track only specified layers
                for name, module in model.named_modules():
                    if any(layer_name in name for layer_name in self.activation_layers):
                        self.activation_hooks.append(module.register_forward_hook(hook_fn(name)))
                        
            self.logger.info(f"Registered activation hooks for {len(self.activation_hooks)} layers")
        except Exception as e:
            self.logger.warning(f"Error registering activation hooks: {e}")

    def after_iter(self, solver):
        """Log metrics after each iteration."""
        if we.rank != 0 or self.wandb_run is None:
            return

        try:
            # Get the outputs from the solver
            outputs = solver.iter_outputs.copy()
            extra_vars = solver.collect_log_vars()
            outputs.update(extra_vars)

            # DEBUG: Print available metrics and loss format
            if solver.total_iter % 10 == 0:
                print(f"WANDB DEBUG - AVAILABLE METRICS: {list(outputs.keys())}")
                print(f"WANDB DEBUG - LOSS FORMAT: {outputs.get('loss', 'NOT FOUND')}")
                if 'loss' in outputs:
                    loss_val = outputs['loss']
                    print(f"WANDB DEBUG - LOSS TYPE: {type(loss_val)}")
                    if isinstance(loss_val, torch.Tensor):
                        print(f"WANDB DEBUG - LOSS TENSOR INFO: shape={loss_val.shape}, dim={loss_val.dim()}, numel={loss_val.numel()}")
                        if loss_val.numel() == 1:
                            print(f"WANDB DEBUG - SCALAR VALUE: {loss_val.item()}")

            # Create a dictionary for wandb metrics
            wandb_metrics = {}
            
            # Add iteration and epoch info
            wandb_metrics["iteration"] = solver.total_iter
            wandb_metrics["epoch"] = solver.epoch

            # Process all metrics from outputs
            for key, value in outputs.items():
                # Special handling for loss values
                if key == 'loss' or key.endswith('_loss'):
                    # Ensure loss is logged as a scalar
                    if isinstance(value, list):
                        # If loss is a list, take the first element (current loss)
                        current_loss = value[0] if len(value) > 0 else None
                        avg_loss = value[1] if len(value) > 1 else None
                        
                        # Convert tensor to scalar if needed
                        if isinstance(current_loss, torch.Tensor):
                            current_loss = current_loss.item()
                        if isinstance(avg_loss, torch.Tensor):
                            avg_loss = avg_loss.item()
                            
                        # Log both current and average loss - use simple keys for main charts
                        if current_loss is not None:
                            # Root level metrics for primary charts 
                            wandb_metrics[key] = current_loss
                            if avg_loss is not None:
                                wandb_metrics[f"{key}_avg"] = avg_loss
                            
                            # Log with mode prefix for organization
                            wandb_metrics[f"{solver.mode}/{key}"] = current_loss
                            wandb_metrics[f"{solver.mode}/{key}/current"] = current_loss
                            
                            # Log exact format shown in console (e.g., "0.0370(0.0917)")
                            if avg_loss is not None:
                                wandb_metrics[f"{solver.mode}/{key}/formatted"] = f"{current_loss:.4f}({avg_loss:.4f})"
                        
                        if avg_loss is not None:
                            wandb_metrics[f"{solver.mode}/{key}/average"] = avg_loss
                    else:
                        # Handle scalar loss
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        
                        # Log loss at root level for primary charts
                        wandb_metrics[key] = value
                        
                        # Also log with prefixes for organization
                        wandb_metrics[f"{solver.mode}/{key}"] = value
                        wandb_metrics[f"{solver.mode}/{key}/current"] = value
                
                # Handle other metrics
                elif isinstance(value, list) and len(value) >= 2:
                    # For metrics with current and average values
                    current_val = value[0]
                    avg_val = value[1]

                    # Convert tensor to Python value if needed
                    if isinstance(current_val, torch.Tensor):
                        current_val = current_val.item()
                    if isinstance(avg_val, torch.Tensor):
                        avg_val = avg_val.item()

                    # Log both current and average values
                    wandb_metrics[f"{key}"] = current_val  # Root level for main charts
                    wandb_metrics[f"{key}_avg"] = avg_val  # Root level for main charts
                    
                    # Also log with hierarchical naming for better organization
                    wandb_metrics[f"{solver.mode}/{key}"] = current_val
                    wandb_metrics[f"{solver.mode}/{key}/current"] = current_val
                    wandb_metrics[f"{solver.mode}/{key}/average"] = avg_val
                    
                    # Formatted version for reference
                    wandb_metrics[f"{solver.mode}/{key}/formatted"] = f"{current_val:.4f}({avg_val:.4f})"
                else:
                    # For simple metrics
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:  # It's a scalar tensor
                            value = value.item()
                        elif key.endswith('_img') and value.dim() in [3, 4]:
                            # It's likely an image tensor, log as image
                            if value.dim() == 4:  # batch of images
                                # Take first image if it's a batch
                                value = value[0]
                            
                            # Normalize if needed
                            if value.max() > 1.0:
                                value = value / 255.0
                            
                            # Convert to wandb Image
                            wandb_metrics[f"{solver.mode}/images/{key}"] = wandb.Image(value)
                            continue  # Skip the normal scalar logging for this key

                    # Log the value
                    if isinstance(value, (int, float, np.number)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        wandb_metrics[key] = value  # Root level for main chart
                        wandb_metrics[f"{solver.mode}/{key}"] = value  # With prefix for organization
            # Log learning rates if available
            if hasattr(solver, 'optimizer') and solver.optimizer is not None:
                for i, param_group in enumerate(solver.optimizer.param_groups):
                    if 'lr' in param_group:
                        wandb_metrics[f"lr/group_{i}"] = param_group['lr']
                        wandb_metrics[f"{solver.mode}/lr/group_{i}"] = param_group['lr']

            # Add system metrics
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
                        wandb_metrics[f"system/gpu{i}/memory_allocated_mb"] = mem_allocated
                        wandb_metrics[f"system/gpu{i}/memory_reserved_mb"] = mem_reserved
                except Exception as e:
                    self.logger.warning(f"Error logging GPU memory: {e}")

            # Always log to wandb regardless of interval for loss metrics
            self.wandb_run.log(wandb_metrics, step=solver.total_iter)
            
            # Only log parameter histograms at the specified interval to avoid excessive data
            if solver.total_iter % self.interval == 0:
                # Track gradients and weights histograms for model parameters
                if solver.is_train_mode and hasattr(solver, 'model') and solver.model is not None:
                    try:
                        # Log parameter histograms
                        param_metrics = {}
                        for name, param in solver.model.named_parameters():
                            if param.requires_grad:
                                # Log parameter values
                                param_metrics[f"model/weights/{name}"] = wandb.Histogram(param.detach().cpu().numpy())
                                
                                # Log parameter gradients if they exist
                                if param.grad is not None:
                                    param_metrics[f"model/gradients/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())
                                
                                # Log gradient norms for each layer
                                if param.grad is not None:
                                    grad_norm = torch.norm(param.grad.detach()).item()
                                    param_metrics[f"model/gradient_norms/{name}"] = grad_norm
                        
                        # Log parameter metrics separately to avoid overwhelming the main metrics
                        if param_metrics:
                            self.wandb_run.log(param_metrics, step=solver.total_iter)
                    except Exception as e:
                        self.logger.warning(f"Error logging parameter histograms: {e}")

            # Log activations if tracking is enabled
            if self.track_activations and solver.total_iter % self.activation_frequency == 0:
                try:
                    activation_metrics = {}
                    for name, activation in self.activations.items():
                        # Skip if activation is None or empty
                        if activation is None or activation.numel() == 0:
                            continue
                            
                        # Log activation statistics
                        if activation.numel() > 0:
                            # Calculate statistics
                            act_mean = activation.mean().item()
                            act_std = activation.std().item()
                            act_min = activation.min().item()
                            act_max = activation.max().item()
                            
                            # Log statistics
                            activation_metrics[f"activations/{name}/mean"] = act_mean
                            activation_metrics[f"activations/{name}/std"] = act_std
                            activation_metrics[f"activations/{name}/min"] = act_min
                            activation_metrics[f"activations/{name}/max"] = act_max
                            
                            # Log histogram
                            activation_metrics[f"activations/{name}/histogram"] = wandb.Histogram(
                                activation.detach().cpu().reshape(-1).numpy()
                            )
                            
                            # Log sparsity (percentage of zeros)
                            zeros = (activation == 0).float().mean().item() * 100
                            activation_metrics[f"activations/{name}/sparsity_pct"] = zeros
                    
                    # Log activation metrics separately
                    if activation_metrics:
                        self.wandb_run.log(activation_metrics, step=solver.total_iter)
                except Exception as e:
                    self.logger.warning(f"Error logging activations: {e}")
            
            # Check for new files in /cache/save_data at every iteration
            self._scan_for_new_files(solver)

        except Exception as e:
            self.logger.warning(f"Error in WandbLogHook.after_iter: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())

    def after_epoch(self, solver):
        """Log metrics after each epoch."""
        if we.rank != 0 or self.wandb_run is None:
            return

        try:
            # Get the outputs from the solver
            outputs = solver.epoch_outputs.copy()

            # Create a dictionary for wandb metrics
            wandb_metrics = {}

            # Process all metrics from outputs
            for key, value in outputs.items():
                # Special handling for loss values
                if key == 'loss' or key.endswith('_loss'):
                    # Ensure loss is logged as a scalar
                    if isinstance(value, list):
                        # If loss is a list, take the first element (current loss)
                        current_loss = value[0] if len(value) > 0 else None
                        avg_loss = value[1] if len(value) > 1 else None
                        
                        # Convert tensor to scalar if needed
                        if isinstance(current_loss, torch.Tensor):
                            current_loss = current_loss.item()
                        if isinstance(avg_loss, torch.Tensor):
                            avg_loss = avg_loss.item()
                            
                        # Log both current and average loss
                        if current_loss is not None:
                            wandb_metrics[f"{solver.mode}/epoch/{key}"] = current_loss
                            wandb_metrics[f"{solver.mode}/{key}/epoch/current"] = current_loss
                        if avg_loss is not None:
                            wandb_metrics[f"{solver.mode}/{key}/epoch/average"] = avg_loss
                    else:
                        # Handle scalar loss
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        wandb_metrics[f"{solver.mode}/epoch/{key}"] = value
                        wandb_metrics[f"{solver.mode}/{key}/epoch"] = value
                
                elif isinstance(value, list) and len(value) >= 2:
                    # For metrics with current and average values
                    current_val = value[0]
                    avg_val = value[1]

                    # Convert tensor to Python value if needed
                    if isinstance(current_val, torch.Tensor):
                        current_val = current_val.item()
                    if isinstance(avg_val, torch.Tensor):
                        avg_val = avg_val.item()

                    # Log both current and average values
                    wandb_metrics[f"{solver.mode}/epoch/{key}"] = current_val
                    wandb_metrics[f"{solver.mode}/epoch/{key}_avg"] = avg_val
                    
                    # Also log with hierarchical naming
                    wandb_metrics[f"{solver.mode}/{key}/epoch/current"] = current_val
                    wandb_metrics[f"{solver.mode}/{key}/epoch/average"] = avg_val
                else:
                    # For simple metrics
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:  # It's a scalar tensor
                            value = value.item()
                        elif key.endswith('_img') and value.dim() in [3, 4]:
                            # It's likely an image tensor, log as image
                            if value.dim() == 4:  # batch of images
                                # Take first image if it's a batch
                                value = value[0]
                            
                            # Normalize if needed
                            if value.max() > 1.0:
                                value = value / 255.0
                            
                            # Convert to wandb Image
                            wandb_metrics[f"{solver.mode}/epoch_images/{key}"] = wandb.Image(value)
                            continue  # Skip the normal scalar logging

                    # Log the value
                    if isinstance(value, (int, float, np.number)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        wandb_metrics[f"{solver.mode}/epoch/{key}"] = value

            # Add epoch summary metrics
            wandb_metrics[f"{solver.mode}/epoch"] = solver.epoch
            wandb_metrics[f"{solver.mode}/epoch_progress"] = solver.epoch / solver.max_epochs * 100 if solver.max_epochs > 0 else 0

            # Log to wandb
            self.wandb_run.log(wandb_metrics, step=solver.total_iter)

            # Log checkpoint as artifact
            if hasattr(solver, 'checkpoint_hook') and solver.checkpoint_hook is not None:
                self._log_checkpoint_artifact(solver)
                
            # Scan for new files in the work directory
            self._scan_for_new_files(solver, force=True)

        except Exception as e:
            self.logger.warning(f"Error in WandbLogHook.after_epoch: {e}")

    def after_solve(self, solver):
        """Log final metrics and artifacts after solving."""
        if we.rank != 0 or self.wandb_run is None:
            return

        try:
            # Create a final summary artifact with all results
            self._create_final_summary_artifact(solver)
            
            # Log final checkpoint as artifact
            if hasattr(solver, 'checkpoint_hook') and solver.checkpoint_hook is not None:
                self._log_checkpoint_artifact(solver, is_final=True)
                
            # Scan for new files in the work directory
            self._scan_for_new_files(solver, force=True, final=True)
            
            # Create and log a final loss plot
            self._create_final_loss_plot(solver)
            
            # Update wandb summary with final metrics
            if hasattr(solver, 'best_val_loss'):
                self.wandb_run.summary.update({"best_val_loss": solver.best_val_loss})
            if hasattr(solver, 'best_val_acc'):
                self.wandb_run.summary.update({"best_val_acc": solver.best_val_acc})
            
            # Log final training time
            if hasattr(solver, 'start_time'):
                training_time = time.time() - solver.start_time
                self.wandb_run.summary.update({
                    "training_time_seconds": training_time,
                    "training_time_formatted": time_since(solver.start_time)
                })
                
            self.logger.info(f"WandbLogHook: Finished logging to wandb run: {self.wandb_run.name}")

        except Exception as e:
            self.logger.warning(f"Error in WandbLogHook.after_solve: {e}")
            
    def _create_final_loss_plot(self, solver):
        """Create and log a final loss plot to wandb.
        
        Args:
            solver: The solver instance
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Get loss history from wandb
            api = wandb.Api()
            run = api.run(f"{self.wandb_run.entity}/{self.wandb_run.project}/{self.wandb_run.id}")
            
            # Extract loss data
            history = run.scan_history()
            iterations = []
            losses = []
            
            # Find all loss metrics
            loss_keys = []
            for row in history:
                for key in row.keys():
                    if 'loss' in key.lower() and key not in loss_keys:
                        loss_keys.append(key)
                break
            
            # Create a plot for each loss metric
            for loss_key in loss_keys:
                iterations = []
                losses = []
                
                for row in history:
                    if loss_key in row and 'iteration' in row:
                        iterations.append(row['iteration'])
                        losses.append(row[loss_key])
                
                if iterations and losses:
                    plt.plot(iterations, losses, label=loss_key)
            
            # Add labels and title
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            
            # Only add legend if there are labeled lines
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend()
            
            # Log the plot to wandb
            self.wandb_run.log({"final_loss_plot": wandb.Image(plt)})
            
            # Close the figure to free memory
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating final loss plot: {e}")

    def _scan_for_new_files(self, solver, force=False, final=False):
        """Scan for new files in the work directory and log them to wandb.
        
        Args:
            solver: The solver instance
            force: Whether to force scanning even if the last scan was recent
            final: Whether this is the final scan at the end of training
        """
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Only scan periodically unless forced
        current_time = time.time()
        if not force and hasattr(self, '_last_scan_time') and current_time - self._last_scan_time < 60:  # 60 seconds
            return
            
        self._last_scan_time = current_time
        
        try:
            # Default work directory to scan
            work_dir = solver.work_dir
            if not os.path.exists(work_dir):
                return
                
            # Track files by extension
            image_extensions = ['.png', '.jpg', '.jpeg']
            video_extensions = ['.mp4', '.gif']
            data_extensions = ['.json', '.yaml', '.csv', '.txt']
            model_extensions = ['.pth', '.pt', '.ckpt', '.bin', '.h5']
            
            # Keep track of files we've already logged
            if not hasattr(self, '_logged_files'):
                self._logged_files = set()
                
            # Create artifacts for different file types if final scan
            if final:
                self._images_artifact = wandb.Artifact(f"images_{self.wandb_run.id}", type="images")
                self._videos_artifact = wandb.Artifact(f"videos_{self.wandb_run.id}", type="videos")
                self._data_artifact = wandb.Artifact(f"data_{self.wandb_run.id}", type="data")
                self._models_artifact = wandb.Artifact(f"models_{self.wandb_run.id}", type="models")
                
            # Walk through the directory
            for root, _, files in os.walk(work_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip if already logged
                    if file_path in self._logged_files:
                        continue
                        
                    # Get file extension
                    _, ext = os.path.splitext(file_path)
                    ext = ext.lower()
                    
                    # Skip certain files
                    if any(pattern in file_path for pattern in ['__pycache__', '.git', 'tmp', 'temp']):
                        continue
                        
                    # Log based on file type
                    try:
                        if ext in image_extensions:
                            # Log image
                            img = wandb.Image(file_path)
                            rel_path = os.path.relpath(file_path, work_dir)
                            self.wandb_run.log({f"files/images/{rel_path}": img}, step=solver.total_iter)
                            
                            if final:
                                self._images_artifact.add_file(file_path, name=rel_path)
                                
                        elif ext in video_extensions:
                            # Log video
                            video = wandb.Video(file_path)
                            rel_path = os.path.relpath(file_path, work_dir)
                            self.wandb_run.log({f"files/videos/{rel_path}": video}, step=solver.total_iter)
                            
                            if final:
                                self._videos_artifact.add_file(file_path, name=rel_path)
                                
                        elif ext in data_extensions:
                            # For data files, try to parse and log metrics if it's JSON
                            if ext == '.json':
                                try:
                                    with open(file_path, 'r') as f:
                                        data = json.load(f)
                                        
                                    # If it contains metrics, log them
                                    if isinstance(data, dict):
                                        metrics = {}
                                        for k, v in data.items():
                                            if isinstance(v, (int, float)):
                                                rel_path = os.path.relpath(file_path, work_dir).replace('/', '_')
                                                metrics[f"files/data/{rel_path}/{k}"] = v
                                                
                                        if metrics:
                                            self.wandb_run.log(metrics, step=solver.total_iter)
                                except:
                                    pass
                                    
                            if final:
                                rel_path = os.path.relpath(file_path, work_dir)
                                self._data_artifact.add_file(file_path, name=rel_path)
                                
                        elif ext in model_extensions:
                            # For model files, just add to artifact
                            if final:
                                rel_path = os.path.relpath(file_path, work_dir)
                                self._models_artifact.add_file(file_path, name=rel_path)
                                
                        # Mark as logged
                        self._logged_files.add(file_path)
                        
                    except Exception as e:
                        self.logger.warning(f"Error logging file {file_path}: {e}")
                        
            # Log artifacts if final
            if final:
                if len(self._images_artifact.manifest.entries) > 0:
                    self.wandb_run.log_artifact(self._images_artifact)
                if len(self._videos_artifact.manifest.entries) > 0:
                    self.wandb_run.log_artifact(self._videos_artifact)
                if len(self._data_artifact.manifest.entries) > 0:
                    self.wandb_run.log_artifact(self._data_artifact)
                if len(self._models_artifact.manifest.entries) > 0:
                    self.wandb_run.log_artifact(self._models_artifact)
                    
        except Exception as e:
            self.logger.warning(f"Error scanning for files: {e}")
            
    def _create_final_summary_artifact(self, solver):
        """Create a final summary artifact with all config and results."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Create summary artifact
            summary_artifact = wandb.Artifact(f"run_summary_{self.wandb_run.id}", type="summary")
            
            # Add config as YAML
            config_path = os.path.join(solver.work_dir, "config_summary.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(solver.cfg, f, default_flow_style=False)
            summary_artifact.add_file(config_path, name="config.yaml")
            
            # Add final metrics
            metrics_path = os.path.join(solver.work_dir, "final_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(solver.epoch_outputs, f, indent=2)
            summary_artifact.add_file(metrics_path, name="final_metrics.json")
            
            # Log the artifact
            self.wandb_run.log_artifact(summary_artifact)
            
        except Exception as e:
            self.logger.warning(f"Error creating final summary artifact: {e}")

    def _log_checkpoint_artifact(self, solver, is_final=False):
        """Log checkpoint as wandb artifact."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Get checkpoint directory
            ckpt_dir = solver.checkpoint_hook.ckpt_dir
            
            # Determine which checkpoint to log
            if is_final:
                ckpt_name = "final"
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pth")
            else:
                ckpt_name = f"epoch_{solver.epoch}"
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pth")
                
            # Check if checkpoint exists
            if not os.path.exists(ckpt_path):
                return
                
            # Create artifact
            artifact = wandb.Artifact(
                name=f"checkpoint_{ckpt_name}",
                type="model",
                description=f"Model checkpoint at {ckpt_name}"
            )
            
            # Add metadata
            artifact.metadata = {
                "epoch": solver.epoch,
                "iteration": solver.total_iter,
                "timestamp": time.time(),
            }
            
            # Add metrics if available
            if hasattr(solver, 'epoch_outputs') and solver.epoch_outputs:
                for k, v in solver.epoch_outputs.items():
                    if isinstance(v, (int, float)) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float))):
                        val = v[0] if isinstance(v, list) else v
                        artifact.metadata[k] = val
                        
            # Add file to artifact
            artifact.add_file(ckpt_path, name=f"{ckpt_name}.pth")
            
            # Log artifact
            self.wandb_run.log_artifact(artifact)
            
        except Exception as e:
            self.logger.warning(f"Error logging checkpoint artifact: {e}")

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
        return dict_to_yaml('WANDB_LOG_HOOK', {
            'PRIORITY': _DEFAULT_LOG_PRIORITY,
            'LOG_DIR': None,
            'PROJECT_NAME': 'scepter-project',
            'RUN_NAME': None,
            'LOG_INTERVAL': 1000,
            'CONFIG_LOGGING': True,
            'SAVE_CODE': True,
            'TAGS': [],
            'ENTITY': None,
            'EARLY_INIT': True,
            'TRACK_ACTIVATIONS': False,
            'ACTIVATION_LAYERS': [],
            'ACTIVATION_FREQUENCY': 100
        })
