# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
import glob
import warnings
import time
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

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

_DEFAULT_TRACKER_PRIORITY = 1100


@HOOKS.register_class()
class WandbFileTrackerHook(Hook):
    """Wandb File Tracker Hook for monitoring and logging files.
    
    This hook watches specified directories for file changes and logs them to wandb.
    Particularly useful for tracking model outputs and results saved to the filesystem.
    """
    
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_TRACKER_PRIORITY,
            'description': 'The priority for processing!'
        },
        'TRACK_INTERVAL': {
            'value': 600,
            'description': 'Interval in seconds to check for new files to track'
        },
        'WATCHED_DIRECTORIES': {
            'value': ['./cache/save_data'],
            'description': 'List of directories to watch for new files'
        },
        'FILE_EXTENSIONS': {
            'value': ['.json', '.yaml', '.png', '.jpg', '.jpeg', '.mp4', '.gif'],
            'description': 'File extensions to track'
        },
        'EXCLUDE_PATTERNS': {
            'value': ['__pycache__', '.git', 'tmp', 'temp'],
            'description': 'Patterns to exclude from tracking'
        },
        'MAX_FILES_PER_SYNC': {
            'value': 50,
            'description': 'Maximum number of files to sync in one interval'
        },
        'CREATE_RESULTS_ARTIFACT': {
            'value': True, 
            'description': 'Whether to create a wandb Artifact with the tracked files'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbFileTrackerHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_TRACKER_PRIORITY)
        self.track_interval = cfg.get('TRACK_INTERVAL', 600)  # Default check every 10 minutes
        self.watched_directories = cfg.get('WATCHED_DIRECTORIES', ['./cache/save_data'])
        self.file_extensions = cfg.get('FILE_EXTENSIONS', 
                                       ['.json', '.yaml', '.png', '.jpg', '.jpeg', '.mp4', '.gif'])
        self.exclude_patterns = cfg.get('EXCLUDE_PATTERNS', ['__pycache__', '.git', 'tmp', 'temp'])
        self.max_files_per_sync = cfg.get('MAX_FILES_PER_SYNC', 50)
        self.create_results_artifact = cfg.get('CREATE_RESULTS_ARTIFACT', True)
        
        # Wandb run reference
        self.wandb_run = None
        
        # Keep track of already processed files
        self.tracked_files = set()
        
        # Last time we checked for new files
        self.last_check_time = time.time()
        
        # Artifact for storing tracked files
        self.results_artifact = None

    def before_solve(self, solver):
        """Connect to existing wandb run and initialize file tracking."""
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run if available
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                solver.logger.warning("WandbFileTrackerHook: No active wandb run found. "
                                     "Make sure to initialize wandb with WandbLogHook before this hook.")
            else:
                solver.logger.info(f"WandbFileTrackerHook: Connected to wandb run: {self.wandb_run.name}")
                
                # Initialize results artifact if needed
                if self.create_results_artifact:
                    self.results_artifact = wandb.Artifact(
                        name=f"model_results_{wandb.run.id}", 
                        type="model_results"
                    )
                
                # Scan directories for existing files to establish baseline
                self._scan_directories(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.before_solve: {e}")

    def after_iter(self, solver):
        """Check for new files periodically during training."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Only check periodically to avoid performance impact
        current_time = time.time()
        if current_time - self.last_check_time < self.track_interval:
            return
            
        try:
            # Update last check time
            self.last_check_time = current_time
            
            # Scan directories for new files
            self._scan_directories(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_iter: {e}")

    def after_epoch(self, solver):
        """Check for new files after each epoch."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Scan directories for new files
            self._scan_directories(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_epoch: {e}")

    def after_solve(self, solver):
        """Final check for new files and log consolidated artifact."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Final scan for new files
            self._scan_directories(solver)
            
            # Log the consolidated artifact if it exists and contains files
            if self.create_results_artifact and self.results_artifact is not None:
                if len(self.results_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.results_artifact)
                    
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_solve: {e}")

    def _scan_directories(self, solver):
        """Scan watched directories for new files and track them in wandb.
        
        Args:
            solver: The solver instance
        """
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Expand relative paths based on solver work directory
            watched_dirs = []
            for dir_path in self.watched_directories:
                # Handle relative paths
                if dir_path.startswith('./'):
                    abs_path = os.path.join(solver.work_dir, dir_path[2:])
                    watched_dirs.append(abs_path)
                else:
                    watched_dirs.append(dir_path)
            
            # Track files in each watched directory
            files_processed = 0
            new_files_found = 0
            
            for directory in watched_dirs:
                # Skip if directory doesn't exist
                if not FS.exists(directory):
                    continue
                
                # List all files in the directory recursively
                for root, dirs, files in self._walk_fs(directory):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
                    
                    # Process files
                    for file in files:
                        # Check extension
                        _, ext = os.path.splitext(file)
                        if ext.lower() not in self.file_extensions:
                            continue
                            
                        # Full path to the file
                        file_path = os.path.join(root, file)
                        
                        # Skip if already tracked
                        if file_path in self.tracked_files:
                            continue
                            
                        # Skip excluded patterns
                        if any(pattern in file_path for pattern in self.exclude_patterns):
                            continue
                            
                        # Track the file
                        self._track_file(file_path, solver)
                        self.tracked_files.add(file_path)
                        
                        new_files_found += 1
                        files_processed += 1
                        
                        # Limit the number of files processed per scan
                        if files_processed >= self.max_files_per_sync:
                            break
                    
                    # Limit the number of files processed per scan
                    if files_processed >= self.max_files_per_sync:
                        break
                        
                # Limit the number of files processed per scan
                if files_processed >= self.max_files_per_sync:
                    break
            
            if new_files_found > 0:
                solver.logger.info(f"WandbFileTrackerHook: Tracked {new_files_found} new files")
                
        except Exception as e:
            solver.logger.warning(f"Error scanning directories: {e}")

    def _walk_fs(self, directory):
        """Walk filesystem recursively, handling both local and remote filesystems.
        
        Args:
            directory: Directory path to walk
            
        Yields:
            Tuples of (root, dirs, files) similar to os.walk
        """
        try:
            # List directory contents
            contents = FS.list_directory(directory)
            
            # Separate directories and files
            dirs = []
            files = []
            
            for item in contents:
                if item.get('is_dir', False):
                    dirs.append(item['relative_path'])
                else:
                    files.append(item['relative_path'])
            
            # Yield current directory contents
            yield directory, dirs, files
            
            # Recursively walk subdirectories
            for dir_name in dirs:
                dir_path = os.path.join(directory, dir_name)
                yield from self._walk_fs(dir_path)
                
        except Exception as e:
            self.logger.warning(f"Error walking directory {directory}: {e}")
            yield directory, [], []

    def _track_file(self, file_path, solver):
        """Track a file in wandb based on its type.
        
        Args:
            file_path: Path to the file to track
            solver: The solver instance
        """
        try:
            # Get file extension and name
            _, ext = os.path.splitext(file_path)
            file_name = os.path.basename(file_path)
            
            # Download the file if it's remote
            with FS.get_from(file_path, wait_finish=True) as local_path:
                # Handle different file types
                if ext.lower() in ['.png', '.jpg', '.jpeg']:
                    # Image file
                    img = wandb.Image(local_path)
                    self.wandb_run.log({f"tracked_files/images/{file_name}": img}, step=solver.total_iter)
                    
                elif ext.lower() in ['.mp4', '.gif']:
                    # Video file
                    video = wandb.Video(local_path)
                    self.wandb_run.log({f"tracked_files/videos/{file_name}": video}, step=solver.total_iter)
                    
                elif ext.lower() in ['.json']:
                    # JSON file
                    try:
                        with open(local_path, 'r') as f:
                            data = json.load(f)
                        
                        # If it's a metrics file, log its contents
                        if any(key in file_name.lower() for key in ['metrics', 'result', 'eval', 'stats']):
                            # Try to extract metrics
                            if isinstance(data, dict):
                                metrics = {}
                                # Flatten simple key-value pairs
                                for k, v in data.items():
                                    if isinstance(v, (int, float)):
                                        metrics[f"tracked_metrics/{k}"] = v
                                
                                if metrics:
                                    self.wandb_run.log(metrics, step=solver.total_iter)
                    except:
                        pass
                
                # Add to artifact if enabled
                if self.create_results_artifact and self.results_artifact is not None:
                    # Get relative path from work directory for better organization
                    rel_path = os.path.relpath(file_path, solver.work_dir)
                    self.results_artifact.add_file(local_path, name=rel_path)
                    
        except Exception as e:
            solver.logger.warning(f"Error tracking file {file_path}: {e}")

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbFileTrackerHook.para_dict,
                            set_name=True)
