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
            'value': 1,  # Check every 1 second for maximum responsiveness
            'description': 'Scan interval for cache/save_data directory'
        },
        'CACHE_SAVE_DATA_PRIORITY': {
            'value': True,
            'description': 'Whether to prioritize scanning the cache/save_data directory'
        },
        'CACHE_SAVE_DATA_ARTIFACT_NAME': {
            'value': 'cache_save_data',
            'description': 'Name prefix for cache/save_data artifacts'
        },
        'CACHE_SAVE_DATA_ARTIFACT_TYPE': {
            'value': 'model_outputs',
            'description': 'Type of artifact for cache/save_data files'
        },
        'CACHE_SAVE_DATA_FILE_TYPES': {
            'value': [
                '.pth', '.pt', '.ckpt', '.bin', '.h5', '.model', '.weights',  # Model files
                '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff',  # Images
                '.mp4', '.avi', '.mov', '.webm',  # Videos
                '.json', '.yaml', '.yml', '.txt', '.csv', '.tsv',  # Data files
                '.npy', '.npz', '.pkl', '.pickle',  # NumPy/pickle files
                '.log', '.out', '.err',  # Log files
                '.py', '.sh', '.md', '.config', '.cfg',  # Config files
                '.pdf', '.html', '.htm',  # Documents
                '*'  # Catch-all to ensure we don't miss anything
            ],
            'description': 'File types to track in cache/save_data directory'
        },
        'FORCE_ARTIFACT_UPLOAD': {
            'value': True,
            'description': 'Whether to force upload artifacts even if they are large'
        },
        'EXTRACT_METRICS_FROM_FILES': {
            'value': True,
            'description': 'Whether to extract metrics from tracked files'
        },
        'METRICS_FILE_PATTERNS': {
            'value': ['*metrics*.json', '*stats*.json', '*results*.json', '*.csv', '*.tsv'],
            'description': 'File patterns to extract metrics from'
        },
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
        
        # Cache/save_data directory specific settings
        self.cache_save_data_scan_interval = cfg.get('CACHE_SAVE_DATA_SCAN_INTERVAL', 1)  # Check every 1 second for maximum responsiveness
        self.cache_save_data_priority = cfg.get('CACHE_SAVE_DATA_PRIORITY', True)
        self.cache_save_data_artifact_name = cfg.get('CACHE_SAVE_DATA_ARTIFACT_NAME', 'cache_save_data')
        self.cache_save_data_artifact_type = cfg.get('CACHE_SAVE_DATA_ARTIFACT_TYPE', 'model_outputs')
        self.cache_save_data_file_types = cfg.get('CACHE_SAVE_DATA_FILE_TYPES', [
            '.pth', '.pt', '.ckpt', '.bin', '.h5', '.model', '.weights',  # Model files
            '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff',  # Images
            '.mp4', '.avi', '.mov', '.webm',  # Videos
            '.json', '.yaml', '.yml', '.txt', '.csv', '.tsv',  # Data files
            '.npy', '.npz', '.pkl', '.pickle',  # NumPy/pickle files
            '.log', '.out', '.err',  # Log files
            '.py', '.sh', '.md', '.config', '.cfg',  # Config files
            '.pdf', '.html', '.htm',  # Documents
            '*'  # Catch-all to ensure we don't miss anything
        ])
        self.force_artifact_upload = cfg.get('FORCE_ARTIFACT_UPLOAD', True)
        
        # Metrics extraction
        self.extract_metrics_from_files = cfg.get('EXTRACT_METRICS_FROM_FILES', True)
        self.metrics_file_patterns = cfg.get('METRICS_FILE_PATTERNS', 
                                            ['*metrics*.json', '*stats*.json', '*results*.json', '*.csv', '*.tsv'])
        self.tracked_metrics = {}
        
        # Wandb run reference
        self.wandb_run = None
        
        # Keep track of already processed files
        self.tracked_files = set()
        
        # Last time we checked for new files
        self.last_check_time = time.time()
        self.last_cache_check_time = time.time()
        
        # Artifact for storing tracked files
        self.results_artifact = None
        
        # Create separate artifacts for different file types
        self.images_artifact = None
        self.videos_artifact = None
        self.data_artifact = None
        self.models_artifact = None
        self.logs_artifact = None
        self.configs_artifact = None
        
        # Special artifacts for cache/save_data
        self.cache_save_data_artifact = None
        
        # Track file counts for reporting
        self.file_counts = defaultdict(int)
        
        # Special attention to cache/save_data directory
        self.cache_save_data_dir = None
        self.cache_save_data_paths = [
            "/app/scepter/cache/save_data",  # Docker container path
            "cache/save_data",               # Relative path
            "./cache/save_data",             # Alternate relative path
            os.path.abspath("cache/save_data")  # Absolute path from current directory
        ]

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
                    
                    # Create special artifact for cache/save_data
                    self.cache_save_data_artifact = wandb.Artifact(
                        name=f"{self.cache_save_data_artifact_name}_{wandb.run.id}",
                        type=self.cache_save_data_artifact_type
                    )
                
                # Discover additional model artifact directories
                self._discover_artifact_directories(solver)
                
                # Initialize cache/save_data directory paths
                self._initialize_cache_save_data_paths(solver)
                
                # Scan directories for existing files to establish baseline
                self._scan_directories(solver)
                
                # Initial scan of cache/save_data directory
                if self.cache_save_data_dir and FS.exists(self.cache_save_data_dir):
                    solver.logger.info(f"WandbFileTrackerHook: Initial scan of cache/save_data directory: {self.cache_save_data_dir}")
                    self._scan_cache_save_data_directory(solver)
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.before_solve: {e}")

    def after_iter(self, solver):
        """Check for new files after each iteration."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Initialize cache/save_data directory if not already done
            if self.cache_save_data_dir is None:
                self._initialize_cache_save_data_paths(solver)
                
            # Always check the cache/save_data directory at every iteration
            # This ensures we capture all artifacts as soon as they're created
            if self.cache_save_data_dir and FS.exists(self.cache_save_data_dir):
                self._scan_cache_save_data_directory(solver)
                
            # Check other directories based on interval or iteration count
            current_time = time.time()
            if solver.total_iter % self.iter_track_frequency == 0 or current_time - self.last_check_time >= self.track_interval:
                # Scan all watched directories
                self._scan_directories(solver)
                self.last_check_time = current_time
                
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_iter: {e}")

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
            if self.cache_save_data_dir and FS.exists(self.cache_save_data_dir):
                solver.logger.info(f"WandbFileTrackerHook: Final scan of cache/save_data directory: {self.cache_save_data_dir}")
                self._scan_cache_save_data_directory(solver)
            
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
                "tracked_files/final_other": self.file_counts['other'],
                "tracked_files/final_cache_save_data": len(self.cache_save_data_artifact.metadata.get("contents", [])) if self.cache_save_data_artifact else 0
            })
                    
        except Exception as e:
            solver.logger.warning(f"Error in WandbFileTrackerHook.after_solve: {e}")
    
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
            
            max_files_reached = False
            for directory in watched_dirs:
                # Skip if directory doesn't exist
                if not FS.exists(directory):
                    continue
                
                # Special handling for cache/save_data directory
                is_cache_dir = self._is_cache_save_data_dir(directory)
                
                # Check if we should scan this directory based on interval
                current_time = time.time()
                if not is_cache_dir:
                    # Use the general interval
                    if current_time - self.last_check_time < self.track_interval:
                        continue
                else:
                    # For cache/save_data directory, use the dedicated interval
                    last_cache_scan_time = getattr(self, 'last_cache_scan_time', 0)
                    cache_interval = getattr(self, 'cache_save_data_scan_interval', self.track_interval)
                    if current_time - last_cache_scan_time < cache_interval:
                        continue
                    # Update the last cache scan time
                    self.last_cache_scan_time = current_time
            
                # Get all files in the directory
                all_files = []
                file_count_before = len(self.tracked_files)
                
                # Walk through the directory
                try:
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
                                
                            # Add to the list of files to process
                            all_files.append(file_path)
                except Exception as e:
                    solver.logger.warning(f"Error walking directory {directory}: {e}")
                
                # Sort files by modification time (newest first)
                try:
                    all_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
                except Exception as e:
                    solver.logger.warning(f"Error sorting files by modification time: {e}")
                
                # Process each file based on whether it's in the cache directory or not
                for file_path in all_files:
                    try:
                        if is_cache_dir:
                            self._track_cache_save_data_file(file_path, solver)
                        else:
                            self._track_file(file_path, solver)
                        
                        # Add to tracked files
                        self.tracked_files.add(file_path)
                        
                        new_files_found += 1
                        files_processed += 1
                        
                        # For cache directory files, log immediately for maximum responsiveness
                        if is_cache_dir and self.cache_save_data_artifact is not None and new_files_found % 5 == 0:
                            try:
                                self._log_artifacts(solver, is_final=False)
                            except Exception as e:
                                solver.logger.warning(f"Error logging artifacts: {e}")
                        
                        # Log progress for large directories
                        if new_files_found % 20 == 0 and solver.logger:
                            solver.logger.debug(f"WandbFileTrackerHook: Tracked {new_files_found} files so far...")
                    except Exception as e:
                        solver.logger.warning(f"Error tracking file {file_path}: {e}")
                    
                    # Limit the number of files processed per scan
                    if files_processed >= max_files:
                        max_files_reached = True
                        break
                
                # Log summary if new files were found
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
                self.cache_save_data_dir and (
                    file_path.startswith(self.cache_save_data_dir) or
                    "/cache/save_data/" in file_path or
                    "\\cache\\save_data\\" in file_path
                )
            )
                
            # For files in cache/save_data, we want to track them even if they don't match our extensions
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

    def _scan_specific_directory(self, directory, solver, force=False):
        """Scan a specific directory for new files.
        
        Args:
            directory: Directory to scan
            solver: The solver instance
            force: Whether to force scanning even if the interval hasn't passed
        """
        try:
            try:
                if not FS.exists(directory):
                    return
            except Exception as e:
                solver.logger.warning(f"Error checking if directory exists {directory}: {e}")
                return
                
            # Special handling for cache/save_data directory
            is_cache_dir = self._is_cache_save_data_dir(directory)
            
            # Check if we should scan this directory based on interval
            current_time = time.time()
            if not force:
                if is_cache_dir:
                    # Use the cache-specific interval
                    if current_time - self.last_cache_check_time < self.cache_save_data_scan_interval:
                        return
                    self.last_cache_check_time = current_time
                else:
                    # Use the general interval
                    if current_time - self.last_check_time < self.track_interval:
                        return
            
            # Get all files in the directory
            all_files = []
            file_count_before = len(self.tracked_files)
            
            # Walk through the directory
            try:
                for root, dirs, files in self._walk_fs(directory):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
                    
                    # Process files
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip if already tracked
                        if file_path in self.tracked_files:
                            continue
                            
                        # Skip excluded patterns
                        if any(pattern in file_path for pattern in self.exclude_patterns):
                            continue
                            
                        # Add to the list of files to process
                        all_files.append(file_path)
            except Exception as e:
                solver.logger.warning(f"Error walking directory {directory}: {e}")
            
            # Sort files by modification time (newest first)
            try:
                all_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            except Exception as e:
                solver.logger.warning(f"Error sorting files by modification time: {e}")
            
            # Track each file
            new_files_found = 0
            for file_path in all_files:
                try:
                    if is_cache_dir:
                        self._track_cache_save_data_file(file_path, solver)
                    else:
                        self._track_file(file_path, solver)
                    
                    self.tracked_files.add(file_path)
                    new_files_found += 1
                    
                    # Log immediately after every few files for maximum responsiveness
                    if new_files_found % 5 == 0:
                        try:
                            self._log_artifacts(solver, is_final=False)
                        except Exception as e:
                            solver.logger.warning(f"Error logging artifacts: {e}")
                except Exception as e:
                    solver.logger.warning(f"Error tracking file {file_path}: {e}")
            
            # Log summary if new files were found
            if new_files_found > 0:
                solver.logger.info(f"WandbFileTrackerHook: Found {new_files_found} new files in {directory}")
                
                # For all directories, log artifacts immediately for maximum visibility
                try:
                    self._log_artifacts(solver, is_final=False)
                except Exception as e:
                    solver.logger.warning(f"Error logging artifacts: {e}")
                
            # Update last check time if not a cache directory (cache time is updated separately)
            if not is_cache_dir:
                self.last_check_time = current_time
                
            # Track scan time
            scan_duration = time.time() - current_time
            if scan_duration > 1.0:  # Only log if scanning took more than 1 second
                solver.logger.debug(f"WandbFileTrackerHook: Scanned {directory} in {scan_duration:.2f}s, found {new_files_found} new files")
        except Exception as e:
            solver.logger.warning(f"Error scanning directory {directory}: {e}")

    def _scan_cache_save_data_directory(self, solver):
        """Dedicated method to scan the cache/save_data directory.
        
        Args:
            solver: The solver instance
        """
        if not self.cache_save_data_dir or not FS.exists(self.cache_save_data_dir):
            return
            
        try:
            # Get all files in the directory
            all_files = []
            file_count_before = len(self.tracked_files)
            
            # Walk through the directory
            try:
                for root, dirs, files in self._walk_fs(self.cache_save_data_dir):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
                    
                    # Process all files in cache/save_data directory
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip if already tracked
                        if file_path in self.tracked_files:
                            continue
                            
                        # For cache/save_data, track all files regardless of extension
                        all_files.append(file_path)
            except Exception as e:
                solver.logger.warning(f"Error walking cache/save_data directory: {e}")
            
            # Sort files by modification time (newest first)
            try:
                all_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            except Exception as e:
                solver.logger.warning(f"Error sorting files by modification time: {e}")
            
            # Track each file
            new_files_found = 0
            for file_path in all_files:
                try:
                    self._track_cache_save_data_file(file_path, solver)
                    self.tracked_files.add(file_path)
                    new_files_found += 1
                except Exception as e:
                    solver.logger.warning(f"Error tracking file {file_path}: {e}")
            
            # Log summary if new files were found
            if new_files_found > 0:
                solver.logger.info(f"WandbFileTrackerHook: Found {new_files_found} new files in cache/save_data directory")
                
                # Log the cache/save_data artifact immediately
                if self.cache_save_data_artifact is not None:
                    try:
                        # Create a unique name for this version of the artifact
                        artifact_name = f"{self.cache_save_data_artifact_name}"
                        if hasattr(solver, 'total_iter'):
                            artifact_name += f"_iter{solver.total_iter}"
                        
                        # Add additional metadata
                        self.cache_save_data_artifact.metadata["logged_at"] = time.time()
                        self.cache_save_data_artifact.metadata["iteration"] = solver.total_iter if hasattr(solver, 'total_iter') else 0
                        self.cache_save_data_artifact.metadata["epoch"] = solver.epoch if hasattr(solver, 'epoch') else 0
                        
                        # Log the artifact
                        self.wandb_run.log_artifact(
                            self.cache_save_data_artifact,
                            name=artifact_name,
                            type=self.cache_save_data_artifact_type
                        )
                        
                        # Create a new artifact for future files
                        self.cache_save_data_artifact = wandb.Artifact(
                            name=self.cache_save_data_artifact_name,
                            type=self.cache_save_data_artifact_type
                        )
                        
                        solver.logger.info(f"WandbFileTrackerHook: Logged cache/save_data artifact with {new_files_found} new files")
                        
                        # Log summary metrics
                        self.wandb_run.log({
                            "cache_save_data/files_count": len(self.tracked_files),
                            "cache_save_data/new_files": new_files_found,
                            "cache_save_data/last_scan": time.time()
                        }, step=solver.total_iter)
                    except Exception as e:
                        solver.logger.warning(f"Error logging cache/save_data artifact: {e}")
                
        except Exception as e:
            solver.logger.warning(f"Error scanning cache/save_data directory: {e}")

    def _track_cache_save_data_file(self, file_path, solver):
        """Track a file from the cache/save_data directory.
        
        Args:
            file_path: Path to the file to track
            solver: The solver instance
        """
        try:
            # Skip if already tracked
            if file_path in self.tracked_files:
                return
                
            # Get file extension and size
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Get file size in MB
            try:
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
            except Exception:
                file_size_mb = 0
                
            # Get relative path for artifact
            try:
                rel_path = os.path.relpath(file_path, self.cache_save_data_dir)
            except ValueError:
                # Fall back to basename if relpath fails
                rel_path = os.path.basename(file_path)
                
            # Get local path for adding to artifact
            try:
                with FS.get_file_to_local(file_path) as local_path:
                    # Add to cache/save_data artifact
                    if self.cache_save_data_artifact is not None:
                        # Mark this file as tracked before adding to artifact
                        # This prevents duplicate tracking if the process takes time
                        self.tracked_files.add(file_path)
                        
                        # Log the new file
                        solver.logger.info(f"Tracking new cache/save_data file: {rel_path} ({file_size_mb:.2f} MB)")
                        
                        # Add to the main cache/save_data artifact
                        self.cache_save_data_artifact.add_file(local_path, name=rel_path)
                        
                        # Add metadata
                        metadata = {
                            "name": rel_path,
                            "path": file_path,
                            "size_mb": file_size_mb,
                            "extension": ext,
                            "iteration": solver.total_iter if hasattr(solver, 'total_iter') else 0,
                            "epoch": solver.epoch if hasattr(solver, 'epoch') else 0,
                            "timestamp": time.time(),
                            "modified_time": os.path.getmtime(local_path)
                        }
                        
                        # Store contents in artifact metadata
                        if "contents" not in self.cache_save_data_artifact.metadata:
                            self.cache_save_data_artifact.metadata["contents"] = []
                        self.cache_save_data_artifact.metadata["contents"].append(metadata)
                        
                        # Extract metrics from file if configured
                        if self.extract_metrics_from_files:
                            self._extract_metrics_from_file(file_path, solver)

        except Exception as e:
            solver.logger.warning(f"Error tracking file {file_path}: {e}")

    def _is_cache_save_data_dir(self, directory):
        """Check if a directory is the cache/save_data directory.
        
        Args:
            directory: Directory to check
            
        Returns:
            bool: True if the directory is the cache/save_data directory
        """
        try:
            if self.cache_save_data_dir is None:
                return False
                
            # Normalize paths for comparison
            norm_dir = os.path.normpath(directory)
            norm_cache_dir = os.path.normpath(self.cache_save_data_dir)
            
            return norm_dir == norm_cache_dir
        except Exception:
            return False

    def _extract_metrics_from_file(self, file_path, solver):
        """Extract metrics from a file and log them to wandb.
        
        Args:
            file_path: Path to the file to extract metrics from
            solver: The solver instance
        """
        try:
            # Skip if file doesn't exist
            if not FS.exists(file_path):
                return
                
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            metrics = {}
            
            # Extract metrics based on file type
            if ext == '.json':
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # If data is a dict, add all scalar values as metrics
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}"] = value
                except Exception as e:
                    solver.logger.warning(f"Error extracting metrics from JSON file {file_path}: {e}")
                    
            elif ext in ['.csv', '.tsv']:
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path, sep=',' if ext == '.csv' else '\t')
                    
                    # If the file has only one row, use column names as keys
                    if len(df) == 1:
                        for col in df.columns:
                            value = df[col].iloc[0]
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{col}"] = value
                except Exception as e:
                    solver.logger.warning(f"Error extracting metrics from CSV/TSV file {file_path}: {e}")
                    
            elif ext in ['.npy', '.npz']:
                try:
                    import numpy as np
                    
                    if ext == '.npy':
                        # For .npy files, load and log statistics
                        data = np.load(file_path)
                        if data.ndim == 1:
                            metrics[f"file_metrics/{os.path.basename(file_path)}/mean"] = float(np.mean(data))
                            metrics[f"file_metrics/{os.path.basename(file_path)}/std"] = float(np.std(data))
                            metrics[f"file_metrics/{os.path.basename(file_path)}/min"] = float(np.min(data))
                            metrics[f"file_metrics/{os.path.basename(file_path)}/max"] = float(np.max(data))
                    else:
                        # For .npz files, load all arrays and log statistics for 1D arrays
                        data_dict = np.load(file_path)
                        for key in data_dict.keys():
                            arr = data_dict[key]
                            if arr.ndim == 1:
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}/mean"] = float(np.mean(arr))
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}/std"] = float(np.std(arr))
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}/min"] = float(np.min(arr))
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}/max"] = float(np.max(arr))
                except Exception as e:
                    solver.logger.warning(f"Error extracting metrics from NPY/NPZ file {file_path}: {e}")
                    
            elif ext in ['.pt', '.pth']:
                try:
                    import torch
                    
                    # Load the PyTorch checkpoint
                    checkpoint = torch.load(file_path, map_location='cpu')
                    
                    # If checkpoint is a dict, extract scalar values
                    if isinstance(checkpoint, dict):
                        for key, value in checkpoint.items():
                            # Check if the value is a scalar tensor or a scalar
                            if isinstance(value, torch.Tensor) and value.numel() == 1:
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}"] = value.item()
                            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                                metrics[f"file_metrics/{os.path.basename(file_path)}/{key}"] = value
                except Exception as e:
                    solver.logger.warning(f"Error extracting metrics from PyTorch checkpoint {file_path}: {e}")
                    
            # Log extracted metrics
            if metrics and self.wandb_run is not None:
                try:
                    step = solver.total_iter if hasattr(solver, 'total_iter') else None
                    self.wandb_run.log(metrics, step=step)
                    solver.logger.info(f"Logged {len(metrics)} metrics from file {os.path.basename(file_path)}")
                except Exception as e:
                    solver.logger.warning(f"Error logging metrics from file {file_path}: {e}")
        except Exception as e:
            solver.logger.warning(f"Error extracting metrics from file {file_path}: {e}")

    def _scan_directory(self, directory, solver, is_cache_dir=False):
        """Scan a directory for new files.
        
        Args:
            directory: Directory to scan
            solver: The solver instance
            is_cache_dir: Whether this is the cache/save_data directory
        """
        try:
            if not FS.exists(directory):
                return
                
            # Track scan time
            scan_start_time = time.time()
            
            # Get all files in the directory
            all_files = []
            file_count_before = len(self.tracked_files)
            
            # Walk through the directory
            try:
                for root, dirs, files in self._walk_fs(directory):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
                    
                    # Process files
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip if already tracked
                        if file_path in self.tracked_files:
                            continue
                            
                        # Skip excluded patterns
                        if any(pattern in file_path for pattern in self.exclude_patterns):
                            continue
                            
                        # Add to the list of files to process
                        all_files.append(file_path)
            except Exception as e:
                solver.logger.warning(f"Error walking directory {directory}: {e}")
            
            # Sort files by modification time (newest first)
            try:
                all_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            except Exception as e:
                solver.logger.warning(f"Error sorting files by modification time: {e}")
            
            # Track each file
            new_files_found = 0
            for file_path in all_files:
                try:
                    if is_cache_dir:
                        self._track_cache_save_data_file(file_path, solver)
                    else:
                        self._track_file(file_path, solver)
                    
                    self.tracked_files.add(file_path)
                    new_files_found += 1
                except Exception as e:
                    solver.logger.warning(f"Error tracking file {file_path}: {e}")
            
            # Log summary if new files were found
            if new_files_found > 0:
                solver.logger.info(f"WandbFileTrackerHook: Found {new_files_found} new files in {directory}")
                
                # For cache directory, log artifacts immediately
                if is_cache_dir and self.cache_save_data_artifact is not None:
                    try:
                        self._log_artifacts(solver)
                    except Exception as e:
                        solver.logger.warning(f"Error logging artifacts: {e}")
                
            # Track scan time
            scan_duration = time.time() - scan_start_time
            if scan_duration > 1.0:  # Only log if scanning took more than 1 second
                solver.logger.debug(f"WandbFileTrackerHook: Scanned {directory} in {scan_duration:.2f}s, found {new_files_found} new files")
            
        except Exception as e:
            solver.logger.warning(f"Error scanning directory {directory}: {e}")
