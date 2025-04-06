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
            'value': 60,
            'description': 'Interval in seconds to check for new files to track'
        },
        'WATCHED_DIRECTORIES': {
            'value': ['./cache/save_data'],
            'description': 'List of directories to watch for new files'
        },
        'FILE_EXTENSIONS': {
            'value': ['.json', '.yaml', '.png', '.jpg', '.jpeg', '.mp4', '.gif', '.pth', '.pt', '.txt', '.csv', '.h5', '.bin'],
            'description': 'File extensions to track'
        },
        'EXCLUDE_PATTERNS': {
            'value': ['__pycache__', '.git', 'tmp', 'temp'],
            'description': 'Patterns to exclude from tracking'
        },
        'MAX_FILES_PER_SYNC': {
            'value': 100,
            'description': 'Maximum number of files to sync in one interval'
        },
        'CREATE_RESULTS_ARTIFACT': {
            'value': True, 
            'description': 'Whether to create a wandb Artifact with the tracked files'
        },
        'TRACK_AFTER_ITER': {
            'value': True,
            'description': 'Whether to track files after each iteration'
        },
        'ITER_TRACK_FREQUENCY': {
            'value': 10,
            'description': 'How often to track files during iterations (every N iterations)'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbFileTrackerHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_TRACKER_PRIORITY)
        self.track_interval = cfg.get('TRACK_INTERVAL', 60)  # Default check every minute
        self.watched_directories = cfg.get('WATCHED_DIRECTORIES', ['./cache/save_data'])
        self.file_extensions = cfg.get('FILE_EXTENSIONS', 
                                       ['.json', '.yaml', '.png', '.jpg', '.jpeg', '.mp4', '.gif', 
                                        '.pth', '.pt', '.txt', '.csv', '.h5', '.bin'])
        self.exclude_patterns = cfg.get('EXCLUDE_PATTERNS', ['__pycache__', '.git', 'tmp', 'temp'])
        self.max_files_per_sync = cfg.get('MAX_FILES_PER_SYNC', 100)
        self.create_results_artifact = cfg.get('CREATE_RESULTS_ARTIFACT', True)
        self.track_after_iter = cfg.get('TRACK_AFTER_ITER', True)
        self.iter_track_frequency = cfg.get('ITER_TRACK_FREQUENCY', 10)
        
        # Wandb run reference
        self.wandb_run = None
        
        # Keep track of already processed files
        self.tracked_files = set()
        
        # Last time we checked for new files
        self.last_check_time = time.time()
        
        # Artifact for storing tracked files
        self.results_artifact = None
        
        # Create separate artifacts for different file types
        self.images_artifact = None
        self.videos_artifact = None
        self.data_artifact = None
        self.models_artifact = None
        
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
                    
                    # Create separate artifacts for different file types
                    self.images_artifact = wandb.Artifact(
                        name=f"images_{wandb.run.id}", 
                        type="images"
                    )
                    
                    self.videos_artifact = wandb.Artifact(
                        name=f"videos_{wandb.run.id}", 
                        type="videos"
                    )
                    
                    self.data_artifact = wandb.Artifact(
                        name=f"data_{wandb.run.id}", 
                        type="data"
                    )
                    
                    self.models_artifact = wandb.Artifact(
                        name=f"models_{wandb.run.id}", 
                        type="models"
                    )
                
                # Scan directories for existing files to establish baseline
                self._scan_directories(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.before_solve: {e}")

    def after_iter(self, solver):
        """Check for new files periodically during training."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Check if we should track after this iteration
        if self.track_after_iter and solver.iter % self.iter_track_frequency == 0:
            try:
                # Scan directories for new files
                self._scan_directories(solver, max_files=10)  # Limit files per iteration to avoid slowdown
            except Exception as e:
                solver.logger.warning(f"Error in WandbFileTrackerHook.after_iter: {e}")
                
        # Only check periodically based on time to avoid performance impact
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

    def _scan_directories(self, solver, max_files=None):
        """Scan watched directories for new files and track them in wandb.
        
        Args:
            solver: The solver instance
            max_files: Maximum number of files to process (overrides self.max_files_per_sync)
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
            max_files = max_files or self.max_files_per_sync
            
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
                        if files_processed >= max_files:
                            break
                    
                    # Limit the number of files processed per scan
                    if files_processed >= max_files:
                        break
                        
                # Limit the number of files processed per scan
                if files_processed >= max_files:
                    break
            
            if new_files_found > 0:
                solver.logger.info(f"WandbFileTrackerHook: Tracked {new_files_found} new files")
                
        except Exception as e:
            solver.logger.warning(f"Error scanning directories: {e}")

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
                    
                    # Add to images artifact
                    if self.images_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.images_artifact.add_file(local_path, name=rel_path)
                    
                elif ext.lower() in ['.mp4', '.gif']:
                    # Video file
                    video = wandb.Video(local_path)
                    self.wandb_run.log({f"tracked_files/videos/{file_name}": video}, step=solver.total_iter)
                    
                    # Add to videos artifact
                    if self.videos_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.videos_artifact.add_file(local_path, name=rel_path)
                    
                elif ext.lower() in ['.json', '.yaml', '.csv', '.txt']:
                    # Data file
                    if ext.lower() == '.json':
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
                    
                    # Add to data artifact
                    if self.data_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.data_artifact.add_file(local_path, name=rel_path)
                
                elif ext.lower() in ['.pth', '.pt', '.ckpt', '.bin', '.h5']:
                    # Model file
                    # Add to models artifact
                    if self.models_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.models_artifact.add_file(local_path, name=rel_path)
                
                # Add to main results artifact if enabled
                if self.create_results_artifact and self.results_artifact is not None:
                    # Get relative path from work directory for better organization
                    rel_path = os.path.relpath(file_path, solver.work_dir)
                    self.results_artifact.add_file(local_path, name=rel_path)
                    
        except Exception as e:
            solver.logger.warning(f"Error tracking file {file_path}: {e}")
            
    def after_epoch(self, solver):
        """Check for new files after each epoch and log artifacts."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Scan directories for new files
            self._scan_directories(solver)
            
            # Log artifacts at the end of each epoch
            self._log_artifacts(solver, is_final=False)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_epoch: {e}")
            
    def after_solve(self, solver):
        """Final check for new files and log consolidated artifact."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Final scan for new files
            self._scan_directories(solver)
            
            # Log all artifacts
            self._log_artifacts(solver, is_final=True)
                    
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_solve: {e}")
            
    def _log_artifacts(self, solver, is_final=False):
        """Log all artifacts to wandb.
        
        Args:
            solver: The solver instance
            is_final: Whether this is the final logging at the end of training
        """
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Log the consolidated artifact if it exists and contains files
            if self.create_results_artifact:
                # Log type-specific artifacts
                if self.images_artifact is not None and len(self.images_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.images_artifact)
                    
                if self.videos_artifact is not None and len(self.videos_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.videos_artifact)
                    
                if self.data_artifact is not None and len(self.data_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.data_artifact)
                    
                if self.models_artifact is not None and len(self.models_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.models_artifact)
                
                # Log main results artifact if it's the final logging
                if is_final and self.results_artifact is not None and len(self.results_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.results_artifact)
                    
        except Exception as e:
            solver.logger.warning(f"Error logging artifacts: {e}")

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

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbFileTrackerHook.para_dict,
                            set_name=True)
