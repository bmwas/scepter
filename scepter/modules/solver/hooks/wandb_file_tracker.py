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
            'value': 60,  # 1 minute interval for time-based tracking
            'description': 'Interval in seconds to check for new files to track'
        },
        'WATCHED_DIRECTORIES': {
            'value': [
                './cache/save_data',  # Main save data directory
                './checkpoints',      # Model checkpoints
                './outputs',          # Generated outputs
                './results',          # Evaluation results
                './logs',             # Log files
                './metrics',          # Metrics files
                './artifacts'         # Other artifacts
            ],
            'description': 'List of directories to watch for new files'
        },
        'FILE_EXTENSIONS': {
            'value': [
                # Model files
                '.pth', '.pt', '.ckpt', '.bin', '.h5', '.model', '.weights',
                # Data files
                '.json', '.yaml', '.yml', '.txt', '.csv', '.tsv', '.npy', '.npz', '.pkl',
                # Image files
                '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff',
                # Video files
                '.mp4', '.avi', '.mov', '.webm',
                # Log files
                '.log', '.out', '.err',
                # Config files
                '.config', '.cfg'
            ],
            'description': 'File extensions to track'
        },
        'EXCLUDE_PATTERNS': {
            'value': ['__pycache__', '.git', 'tmp', 'temp', '.ipynb_checkpoints', 'cache/tmp'],
            'description': 'Patterns to exclude from tracking'
        },
        'MAX_FILES_PER_SYNC': {
            'value': 200,  # Increased from 100 to 200
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
            'value': 1,  # Track at every iteration
            'description': 'How often to track files during iterations (every N iterations)'
        },
        'RECURSIVE_TRACKING': {
            'value': True,
            'description': 'Whether to recursively track files in subdirectories'
        },
        'TRACK_CHECKPOINT_METADATA': {
            'value': True,
            'description': 'Whether to extract and log metadata from checkpoint files'
        },
        'ADDITIONAL_DIRECTORIES': {
            'value': [],
            'description': 'Additional directories to watch (will be combined with WATCHED_DIRECTORIES)'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbFileTrackerHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_TRACKER_PRIORITY)
        self.track_interval = cfg.get('TRACK_INTERVAL', 60)  # Default check every minute
        
        # Combine default and additional directories
        self.watched_directories = cfg.get('WATCHED_DIRECTORIES', [
            './cache/save_data',  # Main save data directory
            './checkpoints',      # Model checkpoints
            './outputs',          # Generated outputs
            './results',          # Evaluation results
            './logs',             # Log files
            './metrics',          # Metrics files
            './artifacts'         # Other artifacts
        ])
        
        # Add any additional directories specified
        additional_dirs = cfg.get('ADDITIONAL_DIRECTORIES', [])
        if additional_dirs:
            self.watched_directories.extend(additional_dirs)
        
        # Ensure the cache/save_data directory is always included
        if './cache/save_data' not in self.watched_directories:
            self.watched_directories.append('./cache/save_data')
        
        self.file_extensions = cfg.get('FILE_EXTENSIONS', [
            # Model files
            '.pth', '.pt', '.ckpt', '.bin', '.h5', '.model', '.weights',
            # Data files
            '.json', '.yaml', '.yml', '.txt', '.csv', '.tsv', '.npy', '.npz', '.pkl',
            # Image files
            '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff',
            # Video files
            '.mp4', '.avi', '.mov', '.webm',
            # Log files
            '.log', '.out', '.err',
            # Config files
            '.config', '.cfg'
        ])
        
        self.exclude_patterns = cfg.get('EXCLUDE_PATTERNS', 
                                       ['__pycache__', '.git', 'tmp', 'temp', '.ipynb_checkpoints', 'cache/tmp'])
        self.max_files_per_sync = cfg.get('MAX_FILES_PER_SYNC', 200)
        self.create_results_artifact = cfg.get('CREATE_RESULTS_ARTIFACT', True)
        self.track_after_iter = cfg.get('TRACK_AFTER_ITER', True)
        self.iter_track_frequency = cfg.get('ITER_TRACK_FREQUENCY', 1)  # Track at every iteration
        self.recursive_tracking = cfg.get('RECURSIVE_TRACKING', True)
        self.track_checkpoint_metadata = cfg.get('TRACK_CHECKPOINT_METADATA', True)
        
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
        self.logs_artifact = None
        self.configs_artifact = None
        
        # Track file counts for reporting
        self.file_counts = {
            'images': 0,
            'videos': 0,
            'data': 0,
            'models': 0,
            'logs': 0,
            'configs': 0,
            'other': 0
        }

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
                    
                    self.logs_artifact = wandb.Artifact(
                        name=f"logs_{wandb.run.id}", 
                        type="logs"
                    )
                    
                    self.configs_artifact = wandb.Artifact(
                        name=f"configs_{wandb.run.id}", 
                        type="configs"
                    )
                
                # Discover additional model artifact directories
                self._discover_artifact_directories(solver)
                
                # Scan directories for existing files to establish baseline
                self._scan_directories(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.before_solve: {e}")

    def after_iter(self, solver):
        """Check for new files at every iteration."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Check for new files at every iteration
        if self.track_after_iter:
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
            
            # Add the work_dir/checkpoints directory explicitly if it exists
            checkpoints_dir = os.path.join(solver.work_dir, 'checkpoints')
            if FS.exists(checkpoints_dir) and checkpoints_dir not in watched_dirs:
                watched_dirs.append(checkpoints_dir)
                
            # Add the cache/save_data directory explicitly if it exists
            cache_dir = os.path.join(solver.work_dir, 'cache/save_data')
            if FS.exists(cache_dir) and cache_dir not in watched_dirs:
                watched_dirs.append(cache_dir)
            
            # Track files in each watched directory
            files_processed = 0
            new_files_found = 0
            max_files = max_files or self.max_files_per_sync
            
            # Log the directories being scanned
            if solver.logger and hasattr(solver.logger, 'debug'):
                solver.logger.debug(f"WandbFileTrackerHook: Scanning directories: {watched_dirs}")
            
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
                        try:
                            self._track_file(file_path, solver)
                            self.tracked_files.add(file_path)
                            
                            new_files_found += 1
                            files_processed += 1
                            
                            # Log progress for large directories
                            if new_files_found % 20 == 0 and solver.logger:
                                solver.logger.debug(f"WandbFileTrackerHook: Tracked {new_files_found} files so far...")
                                
                        except Exception as e:
                            if solver.logger:
                                solver.logger.warning(f"Error tracking file {file_path}: {e}")
                        
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
                # Log summary of tracked files
                file_types_summary = ", ".join([f"{count} {file_type}" for file_type, count in self.file_counts.items() if count > 0])
                solver.logger.info(f"WandbFileTrackerHook: Tracked {new_files_found} new files ({file_types_summary})")
                
                # Log summary to wandb
                self.wandb_run.summary.update({
                    "tracked_files/total": sum(self.file_counts.values()),
                    "tracked_files/images": self.file_counts['images'],
                    "tracked_files/videos": self.file_counts['videos'],
                    "tracked_files/data": self.file_counts['data'],
                    "tracked_files/models": self.file_counts['models'],
                    "tracked_files/logs": self.file_counts['logs'],
                    "tracked_files/configs": self.file_counts['configs'],
                    "tracked_files/other": self.file_counts['other']
                })
                
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
                if ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    # Image file
                    img = wandb.Image(local_path)
                    self.wandb_run.log({f"tracked_files/images/{file_name}": img}, step=solver.total_iter)
                    
                    # Add to images artifact
                    if self.images_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.images_artifact.add_file(local_path, name=rel_path)
                    
                    self.file_counts['images'] += 1
                    
                elif ext.lower() in ['.mp4', '.gif', '.avi', '.mov', '.webm']:
                    # Video file
                    video = wandb.Video(local_path)
                    self.wandb_run.log({f"tracked_files/videos/{file_name}": video}, step=solver.total_iter)
                    
                    # Add to videos artifact
                    if self.videos_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.videos_artifact.add_file(local_path, name=rel_path)
                    
                    self.file_counts['videos'] += 1
                    
                elif ext.lower() in ['.json', '.yaml', '.yml', '.csv', '.tsv', '.txt', '.npy', '.npz', '.pkl']:
                    # Data file
                    if ext.lower() == '.json':
                        try:
                            with open(local_path, 'r') as f:
                                data = json.load(f)
                            
                            # If it's a metrics file, log its contents
                            if any(key in file_name.lower() for key in ['metrics', 'result', 'eval', 'stats', 'score', 'performance']):
                                # Try to extract metrics
                                if isinstance(data, dict):
                                    metrics = {}
                                    # Flatten simple key-value pairs
                                    for k, v in data.items():
                                        if isinstance(v, (int, float)):
                                            metrics[f"tracked_metrics/{k}"] = v
                                        elif isinstance(v, dict):
                                            # Handle one level of nesting
                                            for sub_k, sub_v in v.items():
                                                if isinstance(sub_v, (int, float)):
                                                    metrics[f"tracked_metrics/{k}/{sub_k}"] = sub_v
                                    
                                    if metrics:
                                        self.wandb_run.log(metrics, step=solver.total_iter)
                        except Exception as e:
                            solver.logger.debug(f"Could not parse JSON file {file_path}: {e}")
                    
                    # Add to data artifact
                    if self.data_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.data_artifact.add_file(local_path, name=rel_path)
                    
                    self.file_counts['data'] += 1
                
                elif ext.lower() in ['.pth', '.pt', '.ckpt', '.bin', '.h5', '.model', '.weights']:
                    # Model file - extract metadata if possible
                    if self.track_checkpoint_metadata and ext.lower() in ['.pth', '.pt', '.ckpt']:
                        try:
                            # Try to load metadata without loading the full model
                            checkpoint = torch.load(local_path, map_location='cpu')
                            
                            # Extract metadata if it's a dictionary
                            if isinstance(checkpoint, dict):
                                metadata = {}
                                
                                # Extract common metadata fields
                                for key in ['epoch', 'iter', 'iteration', 'step', 'global_step', 
                                           'best_metric', 'learning_rate', 'optimizer_state']:
                                    if key in checkpoint and isinstance(checkpoint[key], (int, float, str)):
                                        metadata[f"checkpoint/{key}"] = checkpoint[key]
                                
                                # Log metadata if found
                                if metadata:
                                    self.wandb_run.log(metadata, step=solver.total_iter)
                            
                            # Clean up to free memory
                            del checkpoint
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                        except Exception as e:
                            solver.logger.debug(f"Could not extract metadata from checkpoint {file_path}: {e}")
                    
                    # Add to models artifact
                    if self.models_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.models_artifact.add_file(local_path, name=rel_path)
                    
                    self.file_counts['models'] += 1
                
                elif ext.lower() in ['.log', '.out', '.err']:
                    # Log file
                    # Add to logs artifact
                    if self.logs_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.logs_artifact.add_file(local_path, name=rel_path)
                    
                    self.file_counts['logs'] += 1
                
                elif ext.lower() in ['.config', '.cfg']:
                    # Config file
                    # Add to configs artifact
                    if self.configs_artifact is not None:
                        rel_path = os.path.relpath(file_path, solver.work_dir)
                        self.configs_artifact.add_file(local_path, name=rel_path)
                    
                    self.file_counts['configs'] += 1
                
                else:
                    self.file_counts['other'] += 1
                
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
            
            # Log summary statistics
            self.wandb_run.log({
                "epoch": solver.epoch,
                "tracked_files/total_count": sum(self.file_counts.values()),
                "tracked_files/epoch_summary": f"Epoch {solver.epoch}: {sum(self.file_counts.values())} files tracked"
            }, step=solver.total_iter)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_epoch: {e}")
            
    def after_solve(self, solver):
        """Final check for new files and log consolidated artifact."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            solver.logger.info("WandbFileTrackerHook: Performing final scan for model artifacts...")
            
            # Force a thorough scan of all directories with higher max_files limit
            self.max_files_per_sync = 1000  # Increase limit for final scan
            
            # Explicitly scan the cache/save_data directory
            cache_dir = os.path.join(solver.work_dir, 'cache/save_data')
            if FS.exists(cache_dir):
                solver.logger.info(f"WandbFileTrackerHook: Scanning cache directory: {cache_dir}")
                self._scan_specific_directory(cache_dir, solver)
            
            # Explicitly scan the checkpoints directory
            checkpoints_dir = os.path.join(solver.work_dir, 'checkpoints')
            if FS.exists(checkpoints_dir):
                solver.logger.info(f"WandbFileTrackerHook: Scanning checkpoints directory: {checkpoints_dir}")
                self._scan_specific_directory(checkpoints_dir, solver)
            
            # Do one final discovery of artifact directories that might have been created during training
            self._discover_artifact_directories(solver)
            
            # Final scan for new files in all watched directories
            self._scan_directories(solver)
            
            # Log all artifacts
            self._log_artifacts(solver, is_final=True)
            
            # Log final summary
            solver.logger.info(f"WandbFileTrackerHook: Completed tracking with {sum(self.file_counts.values())} total files:")
            for file_type, count in self.file_counts.items():
                if count > 0:
                    solver.logger.info(f"  - {file_type}: {count} files")
                    
            # Update wandb summary with final counts
            self.wandb_run.summary.update({
                "tracked_files/final_total": sum(self.file_counts.values()),
                "tracked_files/final_images": self.file_counts['images'],
                "tracked_files/final_videos": self.file_counts['videos'],
                "tracked_files/final_data": self.file_counts['data'],
                "tracked_files/final_models": self.file_counts['models'],
                "tracked_files/final_logs": self.file_counts['logs'],
                "tracked_files/final_configs": self.file_counts['configs'],
                "tracked_files/final_other": self.file_counts['other']
            })
                    
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_solve: {e}")
    
    def _scan_specific_directory(self, directory, solver):
        """Scan a specific directory for files and track them.
        
        Args:
            directory: The directory to scan
            solver: The solver instance
        """
        if not FS.exists(directory):
            return
            
        try:
            files_tracked = 0
            
            # Walk through the directory
            for root, _, files in self._walk_fs(directory):
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
                    try:
                        self._track_file(file_path, solver)
                        self.tracked_files.add(file_path)
                        files_tracked += 1
                    except Exception as e:
                        solver.logger.warning(f"Error tracking file {file_path}: {e}")
            
            if files_tracked > 0:
                solver.logger.info(f"WandbFileTrackerHook: Tracked {files_tracked} new files from {directory}")
                
        except Exception as e:
            solver.logger.warning(f"Error scanning specific directory {directory}: {e}")
            
    def _log_artifacts(self, solver, is_final=False):
        """Log all artifacts to wandb.
        
        Args:
            solver: The solver instance
            is_final: Whether this is the final logging at the end of training
        """
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Create a new version of artifacts for final logging
            if is_final:
                # Create new artifact versions for final logging
                if self.images_artifact is not None and len(self.images_artifact.metadata.get("contents", [])) > 0:
                    final_images_artifact = wandb.Artifact(
                        name=f"images_final_{wandb.run.id}", 
                        type="images"
                    )
                    
                    # Copy files from the existing artifact
                    for file_info in self.images_artifact.metadata.get("contents", []):
                        if "path" in file_info:
                            final_images_artifact.add_reference(file_info["path"], name=file_info.get("name", ""))
                    
                    # Log the final artifact
                    self.wandb_run.log_artifact(final_images_artifact)
                    
                if self.models_artifact is not None and len(self.models_artifact.metadata.get("contents", [])) > 0:
                    final_models_artifact = wandb.Artifact(
                        name=f"models_final_{wandb.run.id}", 
                        type="models"
                    )
                    
                    # Copy files from the existing artifact
                    for file_info in self.models_artifact.metadata.get("contents", []):
                        if "path" in file_info:
                            final_models_artifact.add_reference(file_info["path"], name=file_info.get("name", ""))
                    
                    # Log the final artifact
                    self.wandb_run.log_artifact(final_models_artifact)
                
                # Log the consolidated results artifact
                if self.results_artifact is not None and len(self.results_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.results_artifact)
                    
                solver.logger.info("WandbFileTrackerHook: Logged final artifacts to wandb")
            else:
                # Log type-specific artifacts
                if self.images_artifact is not None and len(self.images_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.images_artifact)
                    
                if self.videos_artifact is not None and len(self.videos_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.videos_artifact)
                    
                if self.data_artifact is not None and len(self.data_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.data_artifact)
                    
                if self.models_artifact is not None and len(self.models_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.models_artifact)
                
                if self.logs_artifact is not None and len(self.logs_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.logs_artifact)
                
                if self.configs_artifact is not None and len(self.configs_artifact.metadata.get("contents", [])) > 0:
                    self.wandb_run.log_artifact(self.configs_artifact)
                    
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
            # Check if directory exists
            if not FS.exists(directory):
                return
            
            # Get all files and directories in the current directory
            with FS.get_dir_to_local_dir(directory) as local_dir:
                for root, dirs, files in os.walk(local_dir):
                    # Convert local paths back to remote paths
                    remote_root = os.path.join(directory, os.path.relpath(root, local_dir))
                    yield remote_root, dirs, files
                    
                    # If not recursive, don't go into subdirectories
                    if not self.recursive_tracking:
                        break
        except Exception as e:
            # Handle errors gracefully
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Error walking directory {directory}: {e}")
            else:
                warnings.warn(f"Error walking directory {directory}: {e}")

    def _discover_artifact_directories(self, solver):
        """Discover additional directories that might contain model artifacts.
        
        This method looks for common directories where model artifacts might be stored
        in the scepter codebase and adds them to the watched directories list.
        
        Args:
            solver: The solver instance
        """
        try:
            # List of potential artifact directories to check
            potential_dirs = [
                # Common model output directories
                os.path.join(solver.work_dir, 'outputs'),
                os.path.join(solver.work_dir, 'results'),
                os.path.join(solver.work_dir, 'models'),
                os.path.join(solver.work_dir, 'weights'),
                os.path.join(solver.work_dir, 'artifacts'),
                os.path.join(solver.work_dir, 'logs'),
                os.path.join(solver.work_dir, 'metrics'),
                os.path.join(solver.work_dir, 'visualizations'),
                os.path.join(solver.work_dir, 'embeddings'),
                os.path.join(solver.work_dir, 'features'),
                
                # Specific to scepter codebase
                os.path.join(solver.work_dir, 'cache'),
                os.path.join(solver.work_dir, 'cache/save_data'),
                os.path.join(solver.work_dir, 'cache/checkpoints'),
                os.path.join(solver.work_dir, 'cache/models'),
                os.path.join(solver.work_dir, 'cache/results'),
                
                # Method-specific directories
                os.path.join(solver.work_dir, 'methods/scedit/outputs'),
                os.path.join(solver.work_dir, 'methods/studio/outputs'),
            ]
            
            # Check if directories exist and add them to watched_directories
            for dir_path in potential_dirs:
                if FS.exists(dir_path) and dir_path not in self.watched_directories:
                    self.watched_directories.append(dir_path)
                    solver.logger.info(f"WandbFileTrackerHook: Discovered artifact directory: {dir_path}")
            
            # Check if the solver has a specific output directory defined
            if hasattr(solver, 'output_dir') and solver.output_dir:
                output_dir = solver.output_dir
                if FS.exists(output_dir) and output_dir not in self.watched_directories:
                    self.watched_directories.append(output_dir)
                    solver.logger.info(f"WandbFileTrackerHook: Added solver output directory: {output_dir}")
            
            # Check for model-specific directories if solver has a model attribute
            if hasattr(solver, 'model'):
                # If the model has a save_dir attribute
                if hasattr(solver.model, 'save_dir') and solver.model.save_dir:
                    save_dir = solver.model.save_dir
                    if FS.exists(save_dir) and save_dir not in self.watched_directories:
                        self.watched_directories.append(save_dir)
                        solver.logger.info(f"WandbFileTrackerHook: Added model save directory: {save_dir}")
            
            solver.logger.info(f"WandbFileTrackerHook: Watching {len(self.watched_directories)} directories for artifacts")
            
        except Exception as e:
            solver.logger.warning(f"Error discovering artifact directories: {e}")

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbFileTrackerHook.para_dict,
                            set_name=True)
