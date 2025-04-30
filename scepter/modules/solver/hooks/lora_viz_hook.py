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

from scepter.modules.annotator.utils.registry import ANNOTATORS
from scepter.modules.model.registry import MODELS
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.registry import HOOKS
from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.file_system import FS

@HOOKS.register_class()
class LoRAWandbVizHook(Hook):
    """
    Hook to visualize LoRA training progress by generating images at intervals using the current LoRA weights.
    Instead of using predefined prompts, this hook uses the validation dataset for generating images.
    The generated images are logged to wandb for visualization.
    """
    HOOK_PRIORITY = 500  # after validation

    def __init__(self, cfg: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(cfg, logger=logger)
        
        # Configuration
        self.viz_interval = self.cfg.get("VIZ_INTERVAL", 50)  # Generate images every N steps
        self.viz_start = self.cfg.get("VIZ_START", 10)  # Start visualization at step N
        self.num_inference_steps = self.cfg.get("NUM_INFERENCE_STEPS", 20)  # Number of sampling steps
        self.guidance_scale = self.cfg.get("GUIDANCE_SCALE", 4.5)  # Guidance scale for conditional generation
        self.image_size = self.cfg.get("IMAGE_SIZE", 512)  # Output image size
        self.num_val_samples = self.cfg.get("NUM_VAL_SAMPLES", 3)  # Number of validation samples to use

        self.step = 0
        
        self.logger.info(f"🔧 LoRAWandbVizHook: Will generate images every {self.viz_interval} steps, starting at step {self.viz_start}")
        self.logger.info(f"🔧 LoRAWandbVizHook: Using {self.num_inference_steps} sampling steps, guidance scale {self.guidance_scale}")
        
        # Try to get the data loader info from the solver during init
        self.val_data = None

    def before_train(self, *args, **kwargs):
        """Called before training begins"""
        solver = kwargs.get("solver", None)
        if solver is None:
            self.logger.error("❌ LoRAWandbVizHook: Solver not found in kwargs")
            return
        
        # Store solver for later
        self.solver = solver
        
        # Log hook initialization
        self.logger.info("✅ LoRAWandbVizHook initialized")

    def after_iter(self, *args, **kwargs):
        """Called after each training iteration"""
        
        # Get the current step
        solver = kwargs.get("solver", None)
        if solver is None:
            return
        
        self.step = solver.iter
        
        # Check if we should generate images at this step
        if self.step < self.viz_start:
            return
        
        if (self.step == self.viz_start) or (self.step % self.viz_interval == 0):
            self.logger.info("=" * 80)
            self.logger.info(f"📊 LoRAWandbVizHook: STARTING VISUALIZATION at step {self.step}")
            self.logger.info("=" * 80)
            
            # Generate images with current LoRA weights
            self.logger.info(f"🔄 LoRAWandbVizHook: Beginning inference with current LoRA weights...")
            images_generated = self._generate_images_with_current_lora(solver)
            
            if images_generated:
                self.logger.info(f"✅ LoRAWandbVizHook: Successfully generated and logged images at step {self.step}")
            else:
                self.logger.error(f"❌ LoRAWandbVizHook: Failed to generate any images at step {self.step}")

    def _generate_images_with_current_lora(self, solver):
        """
        Generate images using the validation dataset and the current LoRA weights
        """
        # First, get validation data if we don't have it yet
        if not hasattr(solver, 'val_loader') or solver.val_loader is None:
            self.logger.error("❌ LoRAWandbVizHook: No validation data loader found")
            return False
        
        # Get validation dataset info
        val_loader = solver.val_loader
        
        # Get a few samples from the validation set
        val_samples = []
        try:
            # Get an iterator for the validation loader
            val_iter = iter(val_loader)
            # Get the first few samples
            for i in range(min(self.num_val_samples, len(val_loader))):
                sample = next(val_iter)
                val_samples.append(sample)
        except Exception as e:
            self.logger.error(f"❌ LoRAWandbVizHook: Error getting validation samples: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        
        # Store current model mode
        prev_mode = solver.mode
        
        # Switch to test mode
        solver.test_mode()
        self.logger.info(f"🔄 LoRAWandbVizHook: Temporarily switched solver to test mode (was: {prev_mode})")
        
        # Initialize success counter
        successful_images = 0
        
        # List to hold all the image data for logging
        all_log_data = {}
        
        # Process each validation sample
        for i, val_sample in enumerate(val_samples):
            try:
                # Extract data from the validation sample
                # Validation data should have source_img, target_img, and prompt
                
                # Log what we're doing
                self.logger.info(f"🖼️ LoRAWandbVizHook: Processing validation sample {i+1}/{len(val_samples)}")
                
                # Get all keys in the sample
                sample_keys = list(val_sample.keys())
                self.logger.info(f"📄 LoRAWandbVizHook: Sample contains keys: {sample_keys}")
                
                # Extract the prompt and images 
                # Different datasets might have different key names, so we need to check
                prompt = None
                source_img = None
                target_img = None
                
                # Common prompt key names
                prompt_keys = ['prompt', 'text', 'caption']
                for key in prompt_keys:
                    if key in val_sample:
                        prompt = val_sample[key]
                        break
                
                # Common source image key names
                source_keys = ['source_img', 'input_img', 'source', 'input', 'image']
                for key in source_keys:
                    if key in val_sample:
                        source_img = val_sample[key]
                        break
                
                # Common target image key names
                target_keys = ['target_img', 'output_img', 'target', 'output', 'label']
                for key in target_keys:
                    if key in val_sample:
                        target_img = val_sample[key]
                        break
                
                # If we couldn't find the necessary data, skip this sample
                if prompt is None or source_img is None or target_img is None:
                    self.logger.error(f"❌ LoRAWandbVizHook: Couldn't find prompt or images in sample, keys: {sample_keys}")
                    continue
                
                # Prepare the batch data for inference
                batch_data = {}
                
                # The prompt might be a tensor, list, or string, so handle accordingly
                if isinstance(prompt, torch.Tensor):
                    prompt_text = prompt[0] if prompt.dim() > 0 else prompt
                    if isinstance(prompt_text, torch.Tensor) and prompt_text.dtype == torch.int64:
                        # Handle tokenized prompt
                        self.logger.warning("⚠️ LoRAWandbVizHook: Prompt is tokenized, can't use directly for visualization")
                        prompt_text = f"[Tokenized prompt #{i}]"
                    else:
                        prompt_text = str(prompt_text)
                elif isinstance(prompt, list):
                    prompt_text = prompt[0] if len(prompt) > 0 else ""
                else:
                    prompt_text = str(prompt)
                
                self.logger.info(f"📝 LoRAWandbVizHook: Using prompt: '{prompt_text}'")
                
                # Format data according to what the model expects
                batch_data.update({
                    'prompt': [[prompt_text]],    # Nested list format for ACE model
                    'n_prompt': [[""]],           # Empty negative prompt
                    'src_image_list': [[]],       # Required by ACE model even if empty
                    'src_mask_list': [[]],        # Required by ACE model even if empty
                    'sampler': 'ddim',
                    'sample_steps': self.num_inference_steps,
                    'guide_scale': self.guidance_scale,
                    'guide_rescale': 0.5,
                    'image_size': self.image_size,
                    'seed': 42,                   # Fixed seed for reproducibility
                })
                
                # Process source image - ensure it's in the right format for the model
                if isinstance(source_img, torch.Tensor):
                    # Keep just one image if it's a batch
                    if source_img.dim() == 4:  # [B, C, H, W]
                        src_img = source_img[0].detach().clone()  # [C, H, W]
                    else:
                        src_img = source_img.detach().clone()
                else:
                    self.logger.error(f"❌ LoRAWandbVizHook: Source image is not a tensor, type: {type(source_img)}")
                    continue
                
                # Set the image in the batch data
                batch_data['image'] = [src_img]
                
                # Create a mask of all ones (no masking)
                h, w = src_img.shape[-2:]
                batch_data['image_mask'] = [torch.ones(1, h, w, device=src_img.device)]
                
                # Log what we're doing
                self.logger.info(f"🧠 LoRAWandbVizHook: Running inference with keys: {list(batch_data.keys())}")
                
                # Transfer data to cuda
                try:
                    cuda_batch_data = transfer_data_to_cuda(batch_data)
                except Exception as e:
                    self.logger.error(f"❌ LoRAWandbVizHook: Error transferring data to CUDA: {str(e)}")
                    continue
                
                # Run inference
                try:
                    with torch.no_grad():
                        ret = solver.run_step_test(cuda_batch_data)
                        
                    # Extract the generated image
                    gen_img = None
                    for out in ret:
                        if 'image' in out:
                            gen_img = out['image']
                            break
                    
                    if gen_img is None:
                        self.logger.error(f"❌ LoRAWandbVizHook: No 'image' in model output")
                        self.logger.error(f"Available keys in output: {list(out.keys() for out in ret)}")
                        continue
                    
                    # Convert tensor to numpy for wandb logging
                    gen_img_np = gen_img.permute(1, 2, 0).cpu().numpy()
                    gen_img_np = (gen_img_np * 255).astype(np.uint8)
                    
                    # Convert source and target images to numpy as well
                    src_img_np = source_img[0].permute(1, 2, 0).cpu().numpy()
                    src_img_np = (src_img_np * 255).astype(np.uint8)
                    
                    tgt_img_np = target_img[0].permute(1, 2, 0).cpu().numpy()
                    tgt_img_np = (tgt_img_np * 255).astype(np.uint8)
                    
                    # Add to our log data
                    sample_log_data = {
                        f"val_sample_{i+1}/source": wandb.Image(src_img_np, caption="Source"),
                        f"val_sample_{i+1}/target": wandb.Image(tgt_img_np, caption="Target"),
                        f"val_sample_{i+1}/generated": wandb.Image(gen_img_np, caption="Generated"),
                        f"val_sample_{i+1}/prompt": prompt_text
                    }
                    
                    # Add this sample's data to our overall log data
                    all_log_data.update(sample_log_data)
                    
                    # Increment success counter
                    successful_images += 1
                    self.logger.info(f"✅ LoRAWandbVizHook: Successfully generated image {i+1}/{len(val_samples)}")
                    
                except Exception as e:
                    self.logger.error(f"❌ LoRAWandbVizHook: Error during inference: {str(e)}")
                    self.logger.error(traceback.format_exc())
            
            except Exception as e:
                self.logger.error(f"❌ LoRAWandbVizHook: Error processing validation sample: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        # Restore previous mode
        if prev_mode == 'train':
            solver.train_mode()
        elif prev_mode == 'val':
            solver.val_mode()
        elif prev_mode == 'test':
            solver.test_mode()
        self.logger.info(f"🔄 LoRAWandbVizHook: Restored solver to {prev_mode} mode")
        
        # Log all images to wandb
        if successful_images > 0:
            # Add step info to the log data
            all_log_data['step'] = self.step
            
            # Log to wandb
            try:
                wandb.log(all_log_data, step=self.step)
                self.logger.info(f"📊 LoRAWandbVizHook: Logged {successful_images} images to wandb at step {self.step}")
                return True
            except Exception as e:
                self.logger.error(f"❌ LoRAWandbVizHook: Error logging to wandb: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False
        else:
            self.logger.error(f"❌ LoRAWandbVizHook: Failed to generate ANY images from {len(val_samples)} validation samples")
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
