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
from scepter.modules.utils.data import transfer_data_to_cuda

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
                    solver.logger.info("\n" + "="*80)
                    solver.logger.info(f"📊 LoRAWandbVizHook: STARTING VISUALIZATION at step {cur_step}")
                    solver.logger.info("="*80 + "\n")
                    
                    # Generate images using current model state
                    solver.logger.info(f"🔄 LoRAWandbVizHook: Beginning inference with current LoRA weights...")
                    images = self._generate_images_with_current_lora(solver)
                    
                    # Log images to wandb
                    if images:
                        solver.logger.info(f"✅ LoRAWandbVizHook: Successfully generated {len(images)} images!")
                        self._log_images_to_wandb(solver, images, cur_step)
                        self.last_logged_step = cur_step
                    else:
                        solver.logger.error(f"❌ LoRAWandbVizHook: Failed to generate any images at step {cur_step}")
                except Exception as e:
                    solver.logger.error(f"❌ LoRAWandbVizHook FAILED: {str(e)}")
                    solver.logger.error("\n" + "="*80)
                    import traceback
                    solver.logger.error(traceback.format_exc())
                    solver.logger.error("="*80 + "\n")
    
    def _generate_images_with_current_lora(self, solver):
        """Generate images using the current LoRA adapter state."""
        # Initialize return dictionary for images
        result_images = {}
        
        try:
            solver.logger.info(f"📝 LoRAWandbVizHook: Preparing to generate images with {len(self.prompts)} prompts")
            
            # Temporarily set solver to test mode
            saved_mode = solver.mode
            solver.test_mode()
            solver.logger.info(f"🔄 LoRAWandbVizHook: Temporarily switched solver to test mode (was: {saved_mode})")
            
            with torch.no_grad():
                # Process all prompts
                for i, prompt in enumerate(self.prompts):
                    solver.logger.info(f"🖼️ LoRAWandbVizHook: Generating image [{i+1}/{len(self.prompts)}] with prompt: '{prompt}'")
                    
                    # Prepare batch data exactly like in run_inference.py
                    batch_data = {}
                    
                    # Add sample args if available
                    if solver.sample_args:
                        batch_data.update(solver.sample_args.get_lowercase_dict())
                    
                    # Add our specific parameters
                    batch_data.update({
                        'prompt': [prompt],  # Must be a list
                        'n_prompt': [""],    # Empty negative prompt
                        'sampler': 'ddim',
                        'sample_steps': self.num_inference_steps,
                        'guide_scale': self.guidance_scale,
                        'guide_rescale': 0.5,
                        'image_size': self.image_size,
                        'seed': 42,  # Fixed seed for reproducibility
                    })
                    
                    # Log what we're doing
                    solver.logger.info(f"🧠 LoRAWandbVizHook: Running inference with keys: {list(batch_data.keys())}")
                    
                    try:
                        # Run inference exactly like in run_inference.py
                        with torch.autocast(device_type='cuda', enabled=solver.use_amp, dtype=solver.dtype):
                            cuda_batch_data = transfer_data_to_cuda(batch_data)
                            ret = solver.run_step_test(cuda_batch_data)
                            solver.logger.info(f"✅ LoRAWandbVizHook: Generated {len(ret)} images")
                        
                        # Process results - similar to run_inference.py but save to our dictionary
                        for idx, out in enumerate(ret):
                            if 'image' in out:
                                # Convert image to numpy format
                                img = out['image']
                                img_np = img.permute(1, 2, 0).cpu().numpy() * 255
                                img_np = img_np.astype(np.uint8)
                                
                                # Store in our results dictionary
                                key = f"prompt_{i}_sample_{idx}"
                                result_images[key] = {
                                    'image': img_np,
                                    'prompt': prompt
                                }
                                solver.logger.info(f"✅ LoRAWandbVizHook: Added image {key} to results")
                            else:
                                solver.logger.warning(f"⚠️ LoRAWandbVizHook: No 'image' in result[{idx}]")
                                if out:
                                    solver.logger.info(f"ℹ️ LoRAWandbVizHook: Available keys: {list(out.keys())}")
                    
                    except Exception as e:
                        solver.logger.error(f"❌ LoRAWandbVizHook: Error during inference: {e}")
                        import traceback
                        solver.logger.error(traceback.format_exc())
            
            # Restore original mode
            if saved_mode == 'train':
                solver.train_mode()
                solver.logger.info(f"🔄 LoRAWandbVizHook: Restored solver to train mode")
            
        except Exception as e:
            solver.logger.error(f"❌ LoRAWandbVizHook GENERATION ERROR: {str(e)}")
            import traceback
            solver.logger.error(traceback.format_exc())
        
        # Log summary of results
        if result_images:
            solver.logger.info(f"✅ LoRAWandbVizHook: Successfully generated {len(result_images)} images from {len(self.prompts)} prompts")
        else:
            solver.logger.error(f"❌ LoRAWandbVizHook: Failed to generate ANY images from {len(self.prompts)} prompts")
        
        return result_images
    
    def _log_images_to_wandb(self, solver, images, step):
        """Log generated images to wandb."""
        if not images:
            solver.logger.error(f"❌ LoRAWandbVizHook: No images to log to wandb at step {step}")
            return
            
        log_dict = {}
        
        try:
            solver.logger.info(f"📤 LoRAWandbVizHook: Preparing to log {len(images)} images to wandb at step {step}")
            
            # Process each image for wandb logging
            for key, data in images.items():
                img = data['image']
                prompt = data['prompt']
                
                # Create wandb image with caption
                log_dict[f"lora_progress/{key}"] = wandb.Image(
                    img,
                    caption=f"Step {step}: {prompt}"
                )
                solver.logger.info(f"📊 LoRAWandbVizHook: Added image '{key}' to log queue")
            
            # Also create a grid of all images
            if len(images) > 1:
                solver.logger.info(f"🔳 LoRAWandbVizHook: Creating image grid of all {len(images)} images")
                grid_images = [data['image'] for data in images.values()]
                prompts = [data['prompt'] for data in images.values()]
                
                # Log grid as separate image
                log_dict[f"lora_progress/grid"] = wandb.Image(
                    self._create_image_grid(grid_images),
                    caption=f"Step {step} - All prompts"
                )
            
            # Log to wandb
            solver.logger.info(f"📡 LoRAWandbVizHook: Sending {len(log_dict)} images to wandb...")
            self.wandb_run.log(log_dict, step=step)
            solver.logger.info("\n" + "="*80)
            solver.logger.info(f"🎉 LoRAWandbVizHook: SUCCESSFULLY logged {len(images)} images to wandb at step {step}")
            solver.logger.info(f"🔗 LoRAWandbVizHook: Check W&B dashboard at: {self.wandb_run.get_url()}")
            solver.logger.info("="*80 + "\n")
        except Exception as e:
            solver.logger.error(f"❌ LoRAWandbVizHook WANDB LOGGING ERROR: {str(e)}")
            import traceback
            solver.logger.error(traceback.format_exc())
    
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
