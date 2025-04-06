# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import json
import warnings
from typing import Dict, Any, Optional, List

import torch
import numpy as np
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import barrier, we
from scepter.modules.utils.file_system import FS

try:
    import wandb
    from PIL import Image
except Exception as e:
    warnings.warn(f'Running without wandb or PIL! {e}')

_DEFAULT_PROBE_PRIORITY = 1000


@HOOKS.register_class()
class WandbProbeDataHook(Hook):
    """Wandb Probe Data Hook for visualization logging.
    
    Monitors the probe data from the solver and logs images, videos and other 
    visualization data to Weights & Biases.
    """
    
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_PROBE_PRIORITY,
            'description': 'The priority for processing!'
        },
        'PROB_INTERVAL': {
            'value': 1000,
            'description': 'the interval for logging visualizations!'
        },
        'LOG_IMAGES': {
            'value': True,
            'description': 'whether to log images to wandb!'
        },
        'LOG_VIDEOS': {
            'value': True,
            'description': 'whether to log videos to wandb!'
        },
        'IMAGE_UPLOAD_LIMIT': {
            'value': 100,
            'description': 'maximum number of images to upload in a single step!'
        },
        'VIDEO_UPLOAD_LIMIT': {
            'value': 10,
            'description': 'maximum number of videos to upload in a single step!'
        },
        'CREATE_EVALUATION_TABLE': {
            'value': True,
            'description': 'whether to create an evaluation table in wandb!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(WandbProbeDataHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_PROBE_PRIORITY)
        self.prob_interval = cfg.get('PROB_INTERVAL', 1000)
        self.log_images = cfg.get('LOG_IMAGES', True)
        self.log_videos = cfg.get('LOG_VIDEOS', True)
        self.image_upload_limit = cfg.get('IMAGE_UPLOAD_LIMIT', 100)
        self.video_upload_limit = cfg.get('VIDEO_UPLOAD_LIMIT', 10)
        self.create_evaluation_table = cfg.get('CREATE_EVALUATION_TABLE', True)
        
        # Wandb run reference
        self.wandb_run = None
        
        # Track evaluation tables
        self.eval_tables = {}

    def before_solve(self, solver):
        """Connect to existing wandb run if available."""
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run if available
            self.wandb_run = wandb.run
            
            if self.wandb_run is None:
                solver.logger.warning("WandbProbeDataHook: No active wandb run found. "
                                    "Make sure to initialize wandb with WandbLogHook before this hook.")
            else:
                solver.logger.info(f"WandbProbeDataHook: Connected to wandb run: {self.wandb_run.name}")
                
                # Initialize evaluation tables if needed
                if self.create_evaluation_table:
                    # Create evaluation table for storing sample outputs
                    self.eval_tables = {
                        "train": wandb.Table(columns=["step", "epoch", "key", "image", "metadata"]),
                        "val": wandb.Table(columns=["step", "epoch", "key", "image", "metadata"]),
                        "test": wandb.Table(columns=["step", "epoch", "key", "image", "metadata"])
                    }
        except Exception as e:
            solver.logger.warning(f"Error in WandbProbeDataHook.before_solve: {e}")

    def after_iter(self, solver):
        """Log visualization data during training at specified intervals."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Only log during training at specified intervals
        if solver.mode == 'train' and solver.total_iter % self.prob_interval == 0:
            try:
                # Process probe data
                probe_dict = solver.probe_data
                
                if not probe_dict:
                    return
                    
                # Log data to wandb
                self._log_probe_data(
                    solver, 
                    probe_dict, 
                    step=solver.total_iter, 
                    mode=solver.mode,
                    epoch=solver.epoch
                )
                
                # Clear probe data after logging
                solver.clear_probe()
            except Exception as e:
                solver.logger.warning(f"Error in WandbProbeDataHook.after_iter: {e}")

    def after_all_iter(self, solver):
        """Log visualization data after evaluation."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Log for non-training modes (validation, testing)
        if solver.mode != 'train':
            try:
                # Get current training step
                step = solver._total_iter.get('train', 0)
                
                # Process probe data
                probe_dict = solver.probe_data
                
                if not probe_dict:
                    return
                    
                # Log data to wandb
                self._log_probe_data(
                    solver, 
                    probe_dict, 
                    step=step, 
                    mode=solver.mode,
                    epoch=solver.epoch
                )
                
                # Clear probe data after logging
                solver.clear_probe()
            except Exception as e:
                solver.logger.warning(f"Error in WandbProbeDataHook.after_all_iter: {e}")

    def after_solve(self, solver):
        """Log final evaluation tables and artifacts."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        try:
            # Log evaluation tables if they contain data
            for mode, table in self.eval_tables.items():
                if hasattr(table, 'data') and len(table.data) > 0:
                    self.wandb_run.log({f"evaluation/{mode}": table})
        except Exception as e:
            solver.logger.warning(f"Error in WandbProbeDataHook.after_solve: {e}")

    def _log_probe_data(self, solver, probe_dict, step, mode, epoch):
        """Process and log probe data to wandb.
        
        Args:
            solver: The solver instance
            probe_dict: Dictionary of probe data
            step: Current training step
            mode: Current mode (train, val, test)
            epoch: Current epoch
        """
        if not probe_dict:
            return
            
        # Create a dictionary to store all visualizations for wandb logging
        log_dict = {}
        image_count = 0
        video_count = 0
        
        # Create output folder path (we'll use this to save files locally if needed)
        save_folder = os.path.join(
            solver.work_dir,
            f'{mode}_probe/step-{step}'
        )
        
        # Process each key in the probe data
        for key, probe_value in probe_dict.items():
            if not hasattr(probe_value, 'to_log'):
                continue
                
            try:
                # Get the data in a way compatible with original ProbeDataHook
                ret_prefix = os.path.join(
                    save_folder,
                    key.replace('/', '_') + f'_step_{step}'
                )
                
                # Call the probe object's to_log method to get the data
                probe_log_data = probe_value.to_log(
                    ret_prefix, 
                    image_postfix='jpg', 
                    video_postfix='mp4'
                )
                
                if not probe_log_data:
                    continue
                    
                # Process images if available
                if self.log_images and image_count < self.image_upload_limit:
                    images = self._extract_images(probe_log_data)
                    
                    for img_key, img_data in images.items():
                        if isinstance(img_data, str) and img_data.endswith(('.jpg', '.png', '.jpeg')):
                            # It's a file path, try to load it
                            try:
                                with FS.get_from(img_data, wait_finish=True) as local_path:
                                    img = Image.open(local_path)
                                    display_key = f"{mode}/images/{key}/{img_key}"
                                    log_dict[display_key] = wandb.Image(img)
                                    
                                    # Add to evaluation table
                                    if self.create_evaluation_table and mode in self.eval_tables:
                                        self.eval_tables[mode].add_data(
                                            step, 
                                            epoch, 
                                            f"{key}/{img_key}", 
                                            wandb.Image(img),
                                            {
                                                "step": step,
                                                "epoch": epoch,
                                                "mode": mode
                                            }
                                        )
                                    
                                    image_count += 1
                            except Exception as e:
                                solver.logger.warning(f"Failed to load image {img_data}: {e}")
                        elif isinstance(img_data, (np.ndarray, torch.Tensor)):
                            # It's a numpy array or tensor
                            try:
                                if isinstance(img_data, torch.Tensor):
                                    img_data = img_data.detach().cpu().numpy()
                                
                                # Handle different image formats
                                if img_data.ndim == 4:  # Batch of images
                                    for i, single_img in enumerate(img_data):
                                        if i >= self.image_upload_limit:
                                            break
                                        display_key = f"{mode}/images/{key}/{img_key}_{i}"
                                        log_dict[display_key] = wandb.Image(single_img)
                                        image_count += 1
                                elif img_data.ndim == 3:  # Single image
                                    display_key = f"{mode}/images/{key}/{img_key}"
                                    log_dict[display_key] = wandb.Image(img_data)
                                    
                                    # Add to evaluation table
                                    if self.create_evaluation_table and mode in self.eval_tables:
                                        self.eval_tables[mode].add_data(
                                            step, 
                                            epoch, 
                                            f"{key}/{img_key}", 
                                            wandb.Image(img_data),
                                            {
                                                "step": step,
                                                "epoch": epoch,
                                                "mode": mode
                                            }
                                        )
                                        
                                    image_count += 1
                            except Exception as e:
                                solver.logger.warning(f"Failed to process image data: {e}")
                
                # Process videos if available
                if self.log_videos and video_count < self.video_upload_limit:
                    videos = self._extract_videos(probe_log_data)
                    
                    for vid_key, vid_data in videos.items():
                        if isinstance(vid_data, str) and vid_data.endswith(('.mp4', '.gif')):
                            try:
                                with FS.get_from(vid_data, wait_finish=True) as local_path:
                                    display_key = f"{mode}/videos/{key}/{vid_key}"
                                    log_dict[display_key] = wandb.Video(local_path)
                                    video_count += 1
                            except Exception as e:
                                solver.logger.warning(f"Failed to load video {vid_data}: {e}")
                        elif isinstance(vid_data, (np.ndarray, torch.Tensor)):
                            try:
                                if isinstance(vid_data, torch.Tensor):
                                    vid_data = vid_data.detach().cpu().numpy()
                                
                                if vid_data.ndim == 5:  # [batch, frames, channels, height, width]
                                    for i, single_vid in enumerate(vid_data):
                                        if i >= self.video_upload_limit:
                                            break
                                        display_key = f"{mode}/videos/{key}/{vid_key}_{i}"
                                        log_dict[display_key] = wandb.Video(
                                            single_vid, 
                                            fps=4,
                                            format="gif"
                                        )
                                        video_count += 1
                                elif vid_data.ndim == 4:  # [frames, channels, height, width]
                                    display_key = f"{mode}/videos/{key}/{vid_key}"
                                    log_dict[display_key] = wandb.Video(
                                        vid_data, 
                                        fps=4,
                                        format="gif"
                                    )
                                    video_count += 1
                            except Exception as e:
                                solver.logger.warning(f"Failed to process video data: {e}")
                                
                # Process metrics and scalars if available
                metrics = self._extract_metrics(probe_log_data)
                for metric_key, metric_value in metrics.items():
                    log_dict[f"{mode}/metrics/{key}/{metric_key}"] = metric_value
                    
            except Exception as e:
                solver.logger.warning(f"Error processing probe data for key {key}: {e}")
                continue
        
        # Log all data to wandb
        if log_dict:
            self.wandb_run.log(log_dict, step=step)

    def _extract_images(self, data):
        """Extract image data from probe log data.
        
        Args:
            data: The probe log data
            
        Returns:
            Dictionary of image key-value pairs
        """
        images = {}
        
        # Handle dictionary case
        if isinstance(data, dict):
            for k, v in data.items():
                if k.endswith(('image', 'img', 'visualization')):
                    images[k] = v
                elif isinstance(v, dict):
                    # Recursively extract from nested dictionaries
                    nested_images = self._extract_images(v)
                    for nk, nv in nested_images.items():
                        images[f"{k}_{nk}"] = nv
                elif isinstance(v, str) and v.endswith(('.jpg', '.png', '.jpeg')):
                    images[k] = v
                elif isinstance(v, (np.ndarray, torch.Tensor)):
                    # Check if this tensor could be an image
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    else:
                        shape = v.shape
                        
                    # Check tensor dimensions to see if it could be an image
                    if len(shape) in [3, 4] and (shape[-3] in [1, 3, 4] or shape[-1] in [1, 3, 4]):
                        images[k] = v
                        
        # Handle list case (less common)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    nested_images = self._extract_images(item)
                    for nk, nv in nested_images.items():
                        images[f"item_{i}_{nk}"] = nv
                elif isinstance(item, str) and item.endswith(('.jpg', '.png', '.jpeg')):
                    images[f"item_{i}"] = item
                    
        return images

    def _extract_videos(self, data):
        """Extract video data from probe log data.
        
        Args:
            data: The probe log data
            
        Returns:
            Dictionary of video key-value pairs
        """
        videos = {}
        
        # Handle dictionary case
        if isinstance(data, dict):
            for k, v in data.items():
                if k.endswith(('video', 'vid', 'animation', 'sequence')):
                    videos[k] = v
                elif isinstance(v, dict):
                    # Recursively extract from nested dictionaries
                    nested_videos = self._extract_videos(v)
                    for nk, nv in nested_videos.items():
                        videos[f"{k}_{nk}"] = nv
                elif isinstance(v, str) and v.endswith(('.mp4', '.gif')):
                    videos[k] = v
                elif isinstance(v, (np.ndarray, torch.Tensor)):
                    # Check if this tensor could be a video
                    if isinstance(v, torch.Tensor):
                        shape = v.shape
                    else:
                        shape = v.shape
                        
                    # Videos usually have 4+ dimensions
                    if len(shape) >= 4:
                        videos[k] = v
                        
        # Handle list case
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    nested_videos = self._extract_videos(item)
                    for nk, nv in nested_videos.items():
                        videos[f"item_{i}_{nk}"] = nv
                elif isinstance(item, str) and item.endswith(('.mp4', '.gif')):
                    videos[f"item_{i}"] = item
                    
        return videos

    def _extract_metrics(self, data):
        """Extract metrics and scalar values from probe log data.
        
        Args:
            data: The probe log data
            
        Returns:
            Dictionary of metric key-value pairs
        """
        metrics = {}
        
        # Handle dictionary case
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)) and not k.startswith('_'):
                    metrics[k] = v
                elif isinstance(v, dict):
                    # Recursively extract from nested dictionaries
                    nested_metrics = self._extract_metrics(v)
                    for nk, nv in nested_metrics.items():
                        metrics[f"{k}_{nk}"] = nv
                elif isinstance(v, (np.ndarray, torch.Tensor)) and v.size == 1:
                    # Single value tensor/array
                    if isinstance(v, torch.Tensor):
                        metrics[k] = v.item()
                    else:
                        metrics[k] = float(v)
                        
        return metrics

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            WandbProbeDataHook.para_dict,
                            set_name=True)
