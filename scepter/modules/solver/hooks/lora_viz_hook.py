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
        Generate images using direct model inference with the current LoRA weights
        """
        # Store current mode to restore it later
        prev_mode = solver.mode
        
        try:
            # Switch to test mode for inference
            solver.test_mode()
            self.logger.info(f"🔄 LoRAWandbVizHook: Switched to test mode (was: {prev_mode})")
            
            # List to store wandb log data
            log_data = {'step': self.step}
            successful_images = 0
            
            # Use predefined prompts since we don't have validation data
            prompts = [
                "Draw a big house with a pointy roof in a scribble style",
                "Draw a simple house with a doodle",
                "Draw a scribble of a mountain with a long slope"
            ]
            
            # Process each prompt
            for i, prompt_text in enumerate(prompts[:self.num_val_samples]):
                self.logger.info(f"🖼️ LoRAWandbVizHook: Processing prompt {i+1}/{min(len(prompts), self.num_val_samples)}")
                self.logger.info(f"📝 LoRAWandbVizHook: Using prompt: '{prompt_text}'")
                
                try:
                    # Create a blank image as input (just like in the sample code)
                    blank_size = 128 if isinstance(self.image_size, int) and self.image_size > 128 else 64
                    blank_image = torch.zeros(3, blank_size, blank_size, device='cuda')
                    
                    # Prepare inference parameters directly for the model
                    # Following the pattern from the shared sample code
                    try:
                        # Try direct model inference if available
                        if hasattr(solver.model, '__call__'):
                            self.logger.info("🧠 LoRAWandbVizHook: Using direct model inference")
                            results = solver.model(
                                image=[blank_image],
                                mask=[None],
                                task=[""],
                                prompt=[[prompt_text]],
                                negative_prompt=[[""]],
                                output_height=self.image_size if isinstance(self.image_size, int) else self.image_size[0],
                                output_width=self.image_size if isinstance(self.image_size, int) else self.image_size[1],
                                sampler="ddim",
                                sample_steps=self.num_inference_steps,
                                guide_scale=self.guidance_scale,
                                guide_rescale=0.5,
                                seed=42,
                            )
                            
                            if isinstance(results, list) and len(results) > 0:
                                generated_img = results[0]
                                if isinstance(generated_img, torch.Tensor):
                                    gen_np = generated_img.permute(1, 2, 0).cpu().numpy()
                                    gen_np = (gen_np * 255).astype(np.uint8)
                                else:
                                    # If the result is a PIL image
                                    gen_np = np.array(generated_img)
                            else:
                                self.logger.error(f"❌ LoRAWandbVizHook: Unexpected results format: {type(results)}")
                                continue
                        else:
                            # Fallback to solver.run_step_test
                            self.logger.info("🧠 LoRAWandbVizHook: Using solver.run_step_test inference")
                            batch_data = {
                                'prompt': [[prompt_text]],  # Nested list for ACE model
                                'n_prompt': [[""]],         # Empty negative prompt
                                'src_image_list': [[]],     # Required by ACE model
                                'src_mask_list': [[]],      # Required by ACE model
                                'sampler': 'ddim',
                                'sample_steps': self.num_inference_steps,
                                'guide_scale': self.guidance_scale,
                                'guide_rescale': 0.5,
                                'seed': 42,                 # Fixed seed for reproducibility
                                'image': [blank_image],     # Blank image as list
                                'image_mask': [torch.ones(1, blank_size, blank_size, device='cuda')], # Full mask
                            }
                            
                            # Add image size if needed
                            if self.image_size is not None:
                                if isinstance(self.image_size, list):
                                    batch_data['image_size'] = self.image_size
                                else:
                                    batch_data['image_size'] = [self.image_size, self.image_size]
                            
                            # Use same method as in run_inference.py
                            with torch.no_grad():
                                with torch.autocast("cuda", enabled=True, dtype=solver.dtype):
                                    batch_data = transfer_data_to_cuda(batch_data)
                                    results = solver.run_step_test(batch_data)
                            
                            # Extract the generated image
                            generated_img = None
                            for out in results:
                                if 'image' in out:
                                    generated_img = out['image']
                                    break
                                    
                            if generated_img is not None:
                                # Convert to numpy for logging
                                gen_np = generated_img.permute(1, 2, 0).cpu().numpy()
                                gen_np = (gen_np * 255).astype(np.uint8)
                            else:
                                self.logger.error(f"❌ LoRAWandbVizHook: No 'image' in results")
                                continue
                    except Exception as e:
                        self.logger.error(f"❌ LoRAWandbVizHook: Error during inference: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        continue
                    
                    # Create log data for this sample
                    sample_data = {
                        f"lora_viz_{i+1}/generated": wandb.Image(gen_np, caption=f"Generated"),
                        f"lora_viz_{i+1}/prompt": prompt_text
                    }
                    
                    # Add to log data
                    log_data.update(sample_data)
                    successful_images += 1
                    self.logger.info(f"✅ LoRAWandbVizHook: Generated image for prompt {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"❌ LoRAWandbVizHook: Error processing prompt: {str(e)}")
                    self.logger.error(traceback.format_exc())
            
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
