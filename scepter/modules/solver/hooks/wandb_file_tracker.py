# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
import glob
import warnings
import time
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
from collections import defaultdict

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
        },
        'CACHE_SAVE_DATA_SCAN_INTERVAL': {
            'value': 10,  # Check every 10 seconds
            'description': 'Scan interval for cache/save_data directory'
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
        self.file_counts = defaultdict(int)
        
        # Special attention to cache/save_data directory
        self.cache_save_data_dir = None
        self.cache_save_data_paths = [
            "/app/scepter/cache/save_data",  # Docker container path
            "cache/save_data"                # Relative path
        ]
        self.cache_save_data_last_scan_time = 0
        self.cache_save_data_scan_interval = cfg.get('CACHE_SAVE_DATA_SCAN_INTERVAL', 10)  # Check every 10 seconds

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
                
                # Initialize cache/save_data directory paths
                self._initialize_cache_save_data_paths(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.before_solve: {e}")

    def after_iter(self, solver):
        """Check for new files at every iteration."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Initialize cache/save_data directory if not already done
            if self.cache_save_data_dir is None:
                self._initialize_cache_save_data_paths(solver)
            
            # Check if it's time to scan based on iteration count
            if solver.total_iter % self.iter_track_frequency == 0:
                # Check if it's time to scan based on elapsed time
                current_time = time.time()
                elapsed_time = current_time - self.last_check_time
                
                if elapsed_time >= self.track_interval:
                    # Scan all watched directories
                    self._scan_directories(solver)
                    self.last_check_time = current_time
                
                # Always check the cache/save_data directory at higher frequency
                elapsed_time_cache = current_time - self.cache_save_data_last_scan_time
                if self.cache_save_data_dir and FS.exists(self.cache_save_data_dir) and elapsed_time_cache >= self.cache_save_data_scan_interval:
                    solver.logger.debug(f"WandbFileTrackerHook: Scanning cache/save_data directory at iteration {solver.total_iter}")
                    self._scan_specific_directory(self.cache_save_data_dir, solver, recursive=True)
                    self.cache_save_data_last_scan_time = current_time
                    
                    # Log artifacts immediately if new files were found
                    if sum(self.file_counts.values()) > 0:
                        self._log_artifacts(solver, is_final=False)
                        
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
            # Skip if file doesn't exist
            if not FS.exists(file_path):
                return
                
            # Get file extension and type
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Map file to local path if needed
            local_path, _ = FS.map_to_local(file_path)
            
            # Get file size in MB
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            
            # Get relative path for display
            if hasattr(solver, 'work_dir'):
                rel_path = os.path.relpath(file_path, solver.work_dir)
            else:
                rel_path = os.path.basename(file_path)
                
            # Check if this is from cache/save_data directory
            is_cache_save_data = (
                self.cache_save_data_dir and file_path.startswith(self.cache_save_data_dir) or
                file_path.startswith("/app/scepter/cache/save_data") or
                "/cache/save_data/" in file_path
            )
            
            # For cache/save_data, we want to track all files regardless of extension
            # For other directories, we only track files with specific extensions
            if not is_cache_save_data and ext not in self.file_extensions:
                return
            
            # Determine file type and track accordingly
            if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                # Track as image
                if self.images_artifact is not None:
                    try:
                        img = wandb.Image(local_path)
                        self.images_artifact.add_file(local_path, name=rel_path)
                        self.images_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                        self.file_counts['images'] += 1
                        
                        # Log image directly to wandb if it's from cache/save_data
                        if is_cache_save_data:
                            self.wandb_run.log({f"generated_images/{rel_path}": img}, step=solver.total_iter)
                            
                    except Exception as e:
                        solver.logger.warning(f"Error tracking image {file_path}: {e}")
                        
            elif ext in ['.mp4', '.avi', '.mov', '.webm']:
                # Track as video
                if self.videos_artifact is not None:
                    try:
                        self.videos_artifact.add_file(local_path, name=rel_path)
                        self.videos_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                        self.file_counts['videos'] += 1
                        
                        # Log video directly to wandb if it's from cache/save_data
                        if is_cache_save_data:
                            try:
                                video = wandb.Video(local_path)
                                self.wandb_run.log({f"generated_videos/{rel_path}": video}, step=solver.total_iter)
                            except Exception as e:
                                solver.logger.warning(f"Error logging video directly: {e}")
                                
                    except Exception as e:
                        solver.logger.warning(f"Error tracking video {file_path}: {e}")
                        
            elif ext in ['.csv', '.tsv', '.txt', '.json', '.yaml', '.yml', '.h5', '.hdf5', '.pkl', '.pickle']:
                # Track as data file
                if self.data_artifact is not None:
                    try:
                        self.data_artifact.add_file(local_path, name=rel_path)
                        self.data_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                        self.file_counts['data'] += 1
                        
                        # For JSON files from cache/save_data, try to log contents directly
                        if is_cache_save_data and ext == '.json':
                            try:
                                import json
                                with open(local_path, 'r') as f:
                                    json_data = json.load(f)
                                # Log JSON data directly to wandb
                                self.wandb_run.log({f"generated_data/{rel_path}": json_data}, step=solver.total_iter)
                            except Exception as e:
                                solver.logger.warning(f"Error logging JSON data directly: {e}")
                                
                    except Exception as e:
                        solver.logger.warning(f"Error tracking data file {file_path}: {e}")
                        
            elif ext in ['.pth', '.pt', '.ckpt', '.bin']:
                # Track as model file
                if self.models_artifact is not None:
                    try:
                        self.models_artifact.add_file(local_path, name=rel_path)
                        
                        # Try to extract metadata from checkpoint
                        checkpoint_metadata = self._extract_checkpoint_metadata(local_path, solver)
                        
                        self.models_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time(),
                            "metadata": checkpoint_metadata
                        })
                        self.file_counts['models'] += 1
                        
                        # If this is from cache/save_data, log it immediately as a separate artifact
                        if is_cache_save_data:
                            try:
                                checkpoint_artifact = wandb.Artifact(
                                    name=f"checkpoint_{os.path.basename(local_path)}_{solver.total_iter}", 
                                    type="model"
                                )
                                checkpoint_artifact.add_file(local_path, name=os.path.basename(local_path))
                                
                                # Add metadata
                                for key, value in checkpoint_metadata.items():
                                    checkpoint_artifact.metadata[key] = value
                                    
                                # Log the checkpoint artifact immediately
                                self.wandb_run.log_artifact(checkpoint_artifact)
                                solver.logger.info(f"WandbFileTrackerHook: Logged checkpoint artifact for {rel_path}")
                            except Exception as e:
                                solver.logger.warning(f"Error logging checkpoint artifact directly: {e}")
                                
                    except Exception as e:
                        solver.logger.warning(f"Error tracking model file {file_path}: {e}")
                        
            elif ext in ['.log', '.out']:
                # Track as log file
                if self.logs_artifact is not None:
                    try:
                        self.logs_artifact.add_file(local_path, name=rel_path)
                        self.logs_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                        self.file_counts['logs'] += 1
                    except Exception as e:
                        solver.logger.warning(f"Error tracking log file {file_path}: {e}")
                        
            elif ext in ['.py', '.sh', '.md']:
                # Track as config file
                if self.configs_artifact is not None:
                    try:
                        self.configs_artifact.add_file(local_path, name=rel_path)
                        self.configs_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                        self.file_counts['configs'] += 1
                    except Exception as e:
                        solver.logger.warning(f"Error tracking config file {file_path}: {e}")
                        
            else:
                # Track as other file
                if self.results_artifact is not None:
                    try:
                        self.results_artifact.add_file(local_path, name=rel_path)
                        self.results_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                        self.file_counts['other'] += 1
                    except Exception as e:
                        solver.logger.warning(f"Error tracking other file {file_path}: {e}")
                        
            # Add to results artifact if it exists
            if self.results_artifact is not None and self.create_results_artifact:
                try:
                    # Only add to results artifact if not already added
                    if not any(item.get("path") == local_path for item in self.results_artifact.metadata.get("contents", [])):
                        self.results_artifact.add_file(local_path, name=rel_path)
                        self.results_artifact.metadata.setdefault("contents", []).append({
                            "name": rel_path,
                            "path": local_path,
                            "size_mb": file_size_mb,
                            "type": os.path.splitext(file_path)[1],
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "timestamp": time.time()
                        })
                except Exception as e:
                    solver.logger.warning(f"Error adding file to results artifact: {e}")
                    
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
                self._scan_specific_directory(cache_dir, solver, recursive=True)
            
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
    
    def _scan_specific_directory(self, directory, solver, recursive=True, max_files=None):
        """Scan a specific directory for new files.
        
        Args:
            directory: The directory to scan.
            solver: The solver instance.
            recursive: Whether to scan recursively.
            max_files: Maximum number of files to scan.
        """
        try:
            if not FS.exists(directory):
                return
                
            # Check if this is the cache/save_data directory
            is_cache_save_data = (
                self.cache_save_data_dir and directory.startswith(self.cache_save_data_dir) or
                directory == "/app/scepter/cache/save_data" or
                directory.endswith("/cache/save_data")
            )
            
            # Get all files in the directory
            all_files = []
            
            # Walk through the directory
            for root, dirs, files in os.walk(directory):
                # Skip excluded directories
                if any(exclude in root for exclude in self.exclude_patterns):
                    continue
                    
                # Add all files
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip if already tracked
                    if file_path in self.tracked_files:
                        continue
                        
                    # Check file extension if not in cache/save_data
                    _, ext = os.path.splitext(file_path)
                    if not is_cache_save_data and ext.lower() not in self.file_extensions:
                        continue
                        
                    all_files.append(file_path)
                    
                # Stop recursion if not enabled
                if not recursive:
                    break
                    
            # Sort files by modification time (newest first)
            all_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            
            # Limit number of files if specified
            if max_files is not None:
                all_files = all_files[:max_files]
                
            # Track each file
            for file_path in all_files:
                self._track_file(file_path, solver)
                self.tracked_files.add(file_path)
                
            # Log summary
            if all_files:
                solver.logger.info(f"WandbFileTrackerHook: Tracked {len(all_files)} new files in {directory}")
                
        except Exception as e:
            solver.logger.warning(f"Error scanning directory {directory}: {e}")

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

    def _initialize_cache_save_data_paths(self, solver):
        """Initialize the cache/save_data directory paths.
        
        Args:
            solver: The solver instance.
        """
        # Check for /app/scepter/cache/save_data (Docker container path)
        if FS.exists("/app/scepter/cache/save_data"):
            self.cache_save_data_dir = "/app/scepter/cache/save_data"
            if self.cache_save_data_dir not in self.watched_directories:
                self.watched_directories.append(self.cache_save_data_dir)
                solver.logger.info(f"WandbFileTrackerHook: Added Docker cache/save_data directory: {self.cache_save_data_dir}")
            return
            
        # Check for work_dir/cache/save_data
        if hasattr(solver, 'work_dir'):
            work_dir_cache = os.path.join(solver.work_dir, 'cache/save_data')
            if FS.exists(work_dir_cache):
                self.cache_save_data_dir = work_dir_cache
                if self.cache_save_data_dir not in self.watched_directories:
                    self.watched_directories.append(self.cache_save_data_dir)
                    solver.logger.info(f"WandbFileTrackerHook: Added work_dir cache/save_data directory: {self.cache_save_data_dir}")
                return
                
        # Check for relative cache/save_data
        if FS.exists("cache/save_data"):
            self.cache_save_data_dir = os.path.abspath("cache/save_data")
            if self.cache_save_data_dir not in self.watched_directories:
                self.watched_directories.append(self.cache_save_data_dir)
                solver.logger.info(f"WandbFileTrackerHook: Added relative cache/save_data directory: {self.cache_save_data_dir}")
            return
            
        solver.logger.warning("WandbFileTrackerHook: Could not find cache/save_data directory")

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbFileTrackerHook.para_dict,
                            set_name=True)
