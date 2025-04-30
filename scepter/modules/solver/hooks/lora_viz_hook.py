# -*- coding: utf-8 -*-
# Copyright (c) 2023

import copy
import os
import logging
import random
import traceback
from typing import Dict, List, Any, Optional, Union

import numpy as np
import torch
import wandb
from PIL import Image

from scepter.modules.model.registry import MODELS
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.file_system import FS

@HOOKS.register_class()
class LoRAWandbVizHook(Hook):
    """
    Hook to visualize LoRA training progress by generating images at intervals using the current LoRA weights.
    Uses validation dataset samples to show progression of the LoRA adapter training.
    """
    def __init__(self, cfg, logger=None):
        super(LoRAWandbVizHook, self).__init__(cfg, logger=logger)
        
        # Configuration
        self.priority = cfg.get('PRIORITY', 500)
        self.viz_interval = cfg.get('VIZ_INTERVAL', 50)
        self.viz_start = cfg.get('VIZ_START', 5)
        self.num_inference_steps = cfg.get('NUM_INFERENCE_STEPS', 20)
        self.guidance_scale = cfg.get('GUIDANCE_SCALE', 4.5)
        self.image_size = cfg.get('IMAGE_SIZE', 512)
        self.num_val_samples = cfg.get('NUM_VAL_SAMPLES', 3)
        
        self.step = 0
        self.logger.info(f"📋 LoRAWandbVizHook initialized: Will visualize at step {self.viz_start} and every {self.viz_interval} steps after")
    
    def after_iter(self, solver):
        """Generate validation images at specified intervals during training"""
        if solver.mode != 'train':
            return
            
        self.step = solver.iter
        
        # Check if we should generate images at this step
        if self.step < self.viz_start:
            return
            
        if (self.step == self.viz_start) or (self.step % self.viz_interval == 0):
            self.logger.info("=" * 80)
            self.logger.info(f"📊 LoRAWandbVizHook: VISUALIZATION at step {self.step}")
            self.logger.info("=" * 80)
            
            # Generate images with current weights using validation samples
            success = self._visualize_current_lora(solver)
            
            if success:
                self.logger.info(f"✅ LoRAWandbVizHook: Successfully logged images to wandb")
            else:
                self.logger.error(f"❌ LoRAWandbVizHook: Failed to generate images")

    def _visualize_current_lora(self, solver):
        """
        Generate images using the validation dataset and the current LoRA weights
        """
        # Check if validation loader exists
        if not hasattr(solver, 'val_loader') or solver.val_loader is None:
            self.logger.error("❌ LoRAWandbVizHook: No validation loader found")
            return False

        # Store current mode to restore it later
        prev_mode = solver.mode
            
        try:
            # Get validation samples
            val_samples = []
            try:
                # Get iterator for validation loader
                val_iter = iter(solver.val_loader)
                for i in range(min(self.num_val_samples, len(solver.val_loader))):
                    try:
                        sample = next(val_iter)
                        val_samples.append(sample)
                    except StopIteration:
                        break
            except Exception as e:
                self.logger.error(f"❌ LoRAWandbVizHook: Error accessing validation data: {str(e)}")
                return False
                
            if not val_samples:
                self.logger.error("❌ LoRAWandbVizHook: No validation samples obtained")
                return False
                
            # Switch to test mode for inference
            solver.test_mode()
            self.logger.info(f"🔄 LoRAWandbVizHook: Switched to test mode (was: {prev_mode})")
            
            # List to store wandb log data
            log_data = {'step': self.step}
            successful_images = 0
            
            # Process each validation sample
            for i, sample in enumerate(val_samples):
                self.logger.info(f"🖼️ LoRAWandbVizHook: Processing sample {i+1}/{len(val_samples)}")
                
                # Extract data - validation samples should contain images and prompts
                keys = list(sample.keys())
                self.logger.info(f"📄 LoRAWandbVizHook: Sample keys: {keys}")
                
                # Get necessary data from sample
                source_img = None
                target_img = None
                prompt = None
                
                # Find prompt (try common field names)
                for key in ['prompt', 'text', 'caption']:
                    if key in sample:
                        prompt = sample[key]
                        break
                
                # Find source image (try common field names)
                for key in ['source_img', 'source', 'input_img', 'input', 'image']:
                    if key in sample:
                        source_img = sample[key]
                        break
                        
                # Find target image (try common field names)
                for key in ['target_img', 'target', 'output_img', 'output', 'label']:
                    if key in sample:
                        target_img = sample[key]
                        break
                
                # Skip if we're missing data
                if source_img is None or prompt is None:
                    self.logger.error(f"❌ LoRAWandbVizHook: Missing source_img or prompt in sample")
                    continue
                
                # Extract usable prompt
                if isinstance(prompt, torch.Tensor):
                    if prompt.dim() > 0 and prompt.size(0) > 0:
                        if prompt.dtype == torch.int64:
                            # This is probably a tokenized prompt, we can't use it directly
                            prompt_text = f"[Tokenized prompt #{i}]"
                        else:
                            prompt_text = str(prompt[0])
                    else:
                        prompt_text = str(prompt.item() if prompt.numel() == 1 else prompt)
                elif isinstance(prompt, list):
                    prompt_text = prompt[0] if prompt else ""
                else:
                    prompt_text = str(prompt)
                
                self.logger.info(f"📝 LoRAWandbVizHook: Using prompt: '{prompt_text}'")
                
                # Prepare source image
                if isinstance(source_img, torch.Tensor):
                    # Keep one image if it's a batch
                    if source_img.dim() == 4 and source_img.size(0) > 0:  # [B,C,H,W]
                        src_img = source_img[0]
                    else:
                        src_img = source_img
                    
                    # Prepare batch data
                    batch_data = {}
                    
                    # Add sample args if available
                    if hasattr(solver, 'sample_args') and solver.sample_args:
                        batch_data.update(solver.sample_args.get_lowercase_dict())
                    
                    # Format data for ACE model
                    batch_data.update({
                        'prompt': [[prompt_text]],  # Nested list for ACE model
                        'n_prompt': [[""]],         # Empty negative prompt
                        'src_image_list': [[]],     # Required by ACE model
                        'src_mask_list': [[]],      # Required by ACE model
                        'sampler': 'ddim',
                        'sample_steps': self.num_inference_steps,
                        'guide_scale': self.guidance_scale,
                        'guide_rescale': 0.5,
                        'seed': 42,                 # Fixed seed for reproducibility
                        'image': [src_img.detach()], # Source image as list
                        'image_mask': [torch.ones(1, src_img.shape[1], src_img.shape[2], device=src_img.device)], # Full mask
                    })
                    
                    # Add image size if needed
                    if self.image_size is not None:
                        if isinstance(self.image_size, list):
                            batch_data['image_size'] = self.image_size
                        else:
                            batch_data['image_size'] = [self.image_size, self.image_size]
                    
                    # Log what we're doing
                    self.logger.info(f"🧠 LoRAWandbVizHook: Running inference with keys: {list(batch_data.keys())}")
                    
                    # Run inference
                    try:
                        # Transfer data to GPU
                        cuda_batch_data = transfer_data_to_cuda(batch_data)
                        
                        # Use same method as in run_inference.py
                        with torch.no_grad():
                            with torch.autocast("cuda", enabled=True, dtype=solver.dtype):
                                results = solver.run_step_test(cuda_batch_data)
                                
                        # Process results
                        if results:
                            # Find the generated image
                            generated_img = None
                            for out in results:
                                if 'image' in out:
                                    generated_img = out['image']
                                    break
                            
                            if generated_img is not None:
                                # Convert to numpy for logging
                                gen_np = generated_img.permute(1, 2, 0).cpu().numpy()
                                gen_np = (gen_np * 255).astype(np.uint8)
                                
                                # Convert source and target images to numpy
                                src_np = src_img.permute(1, 2, 0).cpu().numpy()
                                src_np = (src_np * 255).astype(np.uint8)
                                
                                # Create log data for this sample
                                sample_data = {
                                    f"val_sample_{i+1}/source": wandb.Image(src_np, caption=f"Source"),
                                    f"val_sample_{i+1}/generated": wandb.Image(gen_np, caption=f"Generated"),
                                    f"val_sample_{i+1}/prompt": prompt_text
                                }
                                
                                # Add target image if available
                                if target_img is not None:
                                    if isinstance(target_img, torch.Tensor):
                                        if target_img.dim() == 4 and target_img.size(0) > 0:
                                            tgt_img = target_img[0]
                                        else:
                                            tgt_img = target_img
                                            
                                        tgt_np = tgt_img.permute(1, 2, 0).cpu().numpy()
                                        tgt_np = (tgt_np * 255).astype(np.uint8)
                                        sample_data[f"val_sample_{i+1}/target"] = wandb.Image(tgt_np, caption=f"Target")
                                
                                # Add to log data
                                log_data.update(sample_data)
                                successful_images += 1
                                self.logger.info(f"✅ LoRAWandbVizHook: Generated image for sample {i+1}")
                            else:
                                self.logger.error(f"❌ LoRAWandbVizHook: No 'image' in results")
                        else:
                            self.logger.error(f"❌ LoRAWandbVizHook: No results from inference")
                    
                    except Exception as e:
                        self.logger.error(f"❌ LoRAWandbVizHook: Error during inference: {str(e)}")
                        self.logger.error(traceback.format_exc())
                else:
                    self.logger.error(f"❌ LoRAWandbVizHook: Source image is not a tensor: {type(source_img)}")
            
            # Restore original mode
            if prev_mode == 'train':
                solver.train_mode()
            elif prev_mode == 'val':
                solver.val_mode()
            self.logger.info(f"🔄 LoRAWandbVizHook: Restored solver to {prev_mode} mode")
            
            # Log to wandb if we have data
            if successful_images > 0:
                try:
                    wandb.log(log_data, step=self.step)
                    self.logger.info(f"📊 LoRAWandbVizHook: Logged {successful_images} images to wandb at step {self.step}")
                    return True
                except Exception as e:
                    self.logger.error(f"❌ LoRAWandbVizHook: Error logging to wandb: {str(e)}")
                    self.logger.error(traceback.format_exc())
            else:
                self.logger.error(f"❌ LoRAWandbVizHook: No successful images to log")
            
            return successful_images > 0
            
        except Exception as e:
            self.logger.error(f"❌ LoRAWandbVizHook: Visualization error: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Restore mode on error
            if prev_mode == 'train':
                solver.train_mode()
            elif prev_mode == 'val':
                solver.val_mode()
            
            return False

def get_config_template():
    return dict_to_yaml({
        'HOOKS': [{
            'NAME': 'LoRAWandbVizHook',
            'PRIORITY': 500,
            'VIZ_INTERVAL': 50,
            'VIZ_START': 10,
            'NUM_INFERENCE_STEPS': 20,
            'GUIDANCE_SCALE': 4.5,
            'IMAGE_SIZE': 512,
            'NUM_VAL_SAMPLES': 3
        }]
    })
