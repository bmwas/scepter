# -*- coding: utf-8 -*-
# Copyright (c) 2023

import os
import torch
import numpy as np
import copy
from tqdm import tqdm
import warnings
from collections import defaultdict

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.model.registry import MODELS
from scepter.modules.utils.general import transfer_data_to_cuda

try:
    import wandb
    from PIL import Image
except ImportError:
    warnings.warn('Running without wandb or PIL!')

# List of prompts to use for visualization
DEFAULT_PROMPTS = [
    "A photo of a beautiful sunset over the ocean",
    "A professional photograph of a majestic white horse running in a green field",
    "An oil painting of a cozy cafe in Paris at night with people sitting outside",
    "A fantasy digital art of a magical castle in the clouds"
]

@HOOKS.register_class()
class LoRAWandbVizHook(Hook):
    """
    Hook to visualize model progress during training by generating images
    with the current state of LoRA adapters at specified intervals.
    
    This hook is designed to:
    1. Generate images using the current LoRA adapters at specified steps
    2. Log these images to wandb to visualize training progress
    3. Not interfere with the normal validation process
    """
    
    para_dict = [{
        'PRIORITY': {
            'value': 500,  # Run after regular steps but before model saving
            'description': 'Priority for processing'
        },
        'VIZ_INTERVAL': {
            'value': 50,
            'description': 'Generate visualization images every N steps'
        },
        'VIZ_START': {
            'value': 10,
            'description': 'Start visualizing at this iteration'
        },
        'PROMPTS': {
            'value': [],
            'description': 'List of prompts to use for image generation'
        },
        'NUM_INFERENCE_STEPS': {
            'value': 20,
            'description': 'Number of inference steps for the diffusion process'
        },
        'GUIDANCE_SCALE': {
            'value': 4.5,
            'description': 'Guidance scale for classifier-free guidance'
        },
        'IMAGE_SIZE': {
            'value': 512,
            'description': 'Size of the generated images'
        }
    }]
    
    def __init__(self, cfg, logger=None):
        super(LoRAWandbVizHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', 500)
        self.viz_interval = cfg.get('VIZ_INTERVAL', 50)
        self.viz_start = cfg.get('VIZ_START', 10)
        self.prompts = cfg.get('PROMPTS', DEFAULT_PROMPTS)
        self.num_inference_steps = cfg.get('NUM_INFERENCE_STEPS', 20)
        self.guidance_scale = cfg.get('GUIDANCE_SCALE', 4.5)
        self.image_size = cfg.get('IMAGE_SIZE', 512)
        
        # Wandb run reference
        self.wandb_run = None
        self.last_logged_step = 0
        
    def before_solve(self, solver):
        """Initialize the hook before training starts."""
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run
            self.wandb_run = wandb.run
            if self.wandb_run is None:
                solver.logger.warning("LoRAWandbVizHook: No active wandb run found")
            else:
                solver.logger.info(f"LoRAWandbVizHook: Connected to wandb run: {self.wandb_run.name}")
                solver.logger.info(f"LoRAWandbVizHook: Will generate images at steps: {self.viz_start}, {self.viz_start + self.viz_interval}, etc.")
        except Exception as e:
            solver.logger.warning(f"Error in LoRAWandbVizHook.before_solve: {e}")
    
    def after_iter(self, solver):
        """Generate and log images at specified intervals."""
        if we.rank != 0 or self.wandb_run is None:
            return
            
        # Only run during training and at specified steps
        if solver.mode == 'train':
            cur_step = solver.total_iter
            should_generate = (
                (cur_step >= self.viz_start) and 
                ((cur_step - self.viz_start) % self.viz_interval == 0 or cur_step == self.viz_start)
            )
            
            if should_generate and cur_step > self.last_logged_step:
                try:
                    solver.logger.info(f"LoRAWandbVizHook: Generating visualization images at step {cur_step}")
                    
                    # Generate images using current model state
                    images = self._generate_images_with_current_lora(solver)
                    
                    # Log images to wandb
                    if images:
                        self._log_images_to_wandb(solver, images, cur_step)
                        self.last_logged_step = cur_step
                except Exception as e:
                    solver.logger.warning(f"Error in LoRAWandbVizHook.after_iter: {e}")
    
    def _generate_images_with_current_lora(self, solver):
        """Generate images using the current LoRA adapter state."""
        # Set model to eval mode temporarily for inference
        solver.model.eval()
        
        # Initialize return dictionary for images
        result_images = {}
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type='cuda', enabled=solver.use_amp, dtype=solver.dtype):
                    # Generate images for each prompt
                    for i, prompt in enumerate(self.prompts):
                        sample_args = {
                            'sampler': 'ddim',
                            'sample_steps': self.num_inference_steps,
                            'guide_scale': self.guidance_scale,
                            'guide_rescale': 0.5,
                        }
                        
                        # Create inference batch data
                        batch_data = {
                            'instruction': prompt,
                            'image_size': self.image_size,
                        }
                        batch_data.update(sample_args)
                        
                        # Run inference
                        result = solver.model.inference(transfer_data_to_cuda(batch_data))
                        
                        # Extract generated image
                        if 'edit_image' in result and result['edit_image'] is not None:
                            for j, edit_img in enumerate(result['edit_image']):
                                if edit_img is not None:
                                    # Convert to numpy uint8 format (0-255)
                                    img_np = (edit_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                    result_images[f"prompt_{i}_{j}"] = {
                                        'image': img_np,
                                        'prompt': prompt
                                    }
        except Exception as e:
            solver.logger.warning(f"Error generating images with LoRA: {e}")
        
        # Restore model to training mode
        solver.model.train()
        
        return result_images
    
    def _log_images_to_wandb(self, solver, images, step):
        """Log generated images to wandb."""
        if not images:
            return
            
        log_dict = {}
        
        try:
            # Process each image for wandb logging
            for key, data in images.items():
                img = data['image']
                prompt = data['prompt']
                
                # Create wandb image with caption
                log_dict[f"lora_progress/{key}"] = wandb.Image(
                    img,
                    caption=f"Step {step}: {prompt}"
                )
            
            # Also create a grid of all images
            if len(images) > 1:
                grid_images = [data['image'] for data in images.values()]
                prompts = [data['prompt'] for data in images.values()]
                
                # Log grid as separate image
                log_dict[f"lora_progress/grid"] = wandb.Image(
                    self._create_image_grid(grid_images),
                    caption=f"Step {step} - All prompts"
                )
            
            # Log to wandb
            self.wandb_run.log(log_dict, step=step)
            solver.logger.info(f"Logged {len(images)} LoRA visualization images to wandb at step {step}")
        except Exception as e:
            solver.logger.warning(f"Error logging images to wandb: {e}")
    
    def _create_image_grid(self, images, cols=2):
        """Create a grid of images for visualization."""
        if not images:
            return None
            
        rows = (len(images) + cols - 1) // cols
        grid_height = rows * images[0].shape[0]
        grid_width = cols * images[0].shape[1]
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid[row*img.shape[0]:(row+1)*img.shape[0], col*img.shape[1]:(col+1)*img.shape[1]] = img
            
        return grid

def get_config_template():
    return dict_to_yaml({
        'HOOKS': [{
            'NAME': 'LoRAWandbVizHook',
            'PRIORITY': 500,
            'VIZ_INTERVAL': 50,
            'VIZ_START': 10,
            'PROMPTS': DEFAULT_PROMPTS,
            'NUM_INFERENCE_STEPS': 20,
            'GUIDANCE_SCALE': 4.5,
            'IMAGE_SIZE': 512
        }]
    })
