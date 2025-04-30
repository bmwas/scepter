# -*- coding: utf-8 -*-
# Copyright (c) 2023

import copy
import os
import logging
import random
import traceback
import csv
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
        
        # CSV validation data info (will be populated from solver config)
        self.csv_path = None
        self.image_root_dir = None
        self.val_samples = None
        
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
            
            # Load validation data from CSV
            val_data = self._load_validation_data(solver)
            
            # Process each validation sample
            for i, sample in enumerate(val_data[:self.num_val_samples]):
                self.logger.info(f"🖼️ LoRAWandbVizHook: Processing sample {i+1}/{min(len(val_data), self.num_val_samples)}")
                self.logger.info(f"📝 LoRAWandbVizHook: Using prompt: '{sample['prompt']}'")
                
                try:
                    # Prepare inference parameters directly for the model
                    # Following the pattern from the shared sample code
                    self.logger.info("🧠 LoRAWandbVizHook: Using solver.run_step_test for inference")
                    
                    # Get the device from the solver
                    device = next(solver.model.parameters()).device
                    
                    # Move input data to the right device
                    source_image = sample['image'].to(device)
                    mask = sample['mask'].to(device)
                    
                    # Log shape information for debugging
                    self.logger.info(f"🧠 LoRAWandbVizHook: Input image shape: {source_image.shape}, device: {source_image.device}")
                    
                    # Create the input data exactly according to ACE model requirements
                    # The key is ensuring src_image_list, src_mask_list, and prompt all have the same length and structure
                    batch_data = {
                        'prompt': [[sample['prompt']]],  # Nested list format [[prompt1]]
                        'n_prompt': [[""]],               # Empty negative prompt in same format
                        'src_image_list': [[source_image]],  # Matching structure [[img1]]
                        'src_mask_list': [[mask]],          # Matching structure [[mask1]]
                        'sampler': 'ddim',
                        'sample_steps': self.num_inference_steps,
                        'guide_scale': self.guidance_scale,
                        'show_process': False,
                        'seed': 42,
                        'image': [source_image],  # Regular image parameter
                        'image_mask': [mask],     # Regular mask parameter
                    }
                    
                    # Add image size if needed
                    if self.image_size is not None:
                        if isinstance(self.image_size, list):
                            batch_data['image_size'] = self.image_size
                        else:
                            batch_data['image_size'] = [self.image_size, self.image_size]
                    
                    # Add batch_data to a list because that's what run_step_test expects
                    batch_list = [batch_data]
                    
                    # Transfer the batch data to CUDA and prepare it for the model
                    batch_list = transfer_data_to_cuda([batch_data])
                    
                    # Run inference using solver.run_step_test
                    with torch.no_grad():
                        with torch.autocast("cuda"):
                            try:
                                result = solver.run_step_test(batch_list, return_output=True)
                                self.logger.info(f"✅ LoRAWandbVizHook: Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                            except Exception as e:
                                self.logger.error(f"❌ LoRAWandbVizHook: Run step test failed: {str(e)}")
                                self.logger.error(traceback.format_exc())
                                continue
                    
                    # Check for valid output - try different possible output key names
                    output_image = None
                    possible_keys = ['image', 'images', 'gen_imgs', 'generated_images', 'output']
                    
                    for key in possible_keys:
                        if key in result and result[key] is not None:
                            if isinstance(result[key], list) and len(result[key]) > 0:
                                output_image = result[key][0]
                                self.logger.info(f"✅ LoRAWandbVizHook: Found image in result['{key}']")
                                break
                            elif isinstance(result[key], torch.Tensor):
                                output_image = result[key]
                                self.logger.info(f"✅ LoRAWandbVizHook: Found tensor in result['{key}']")
                                break
                    
                    # Check if we have a valid output image
                    if output_image is not None:
                        # Get the image from the result
                        generated_img = output_image
                        
                        # Convert to numpy for logging
                        gen_np = generated_img.permute(1, 2, 0).cpu().numpy()
                        gen_np = (gen_np * 255).astype(np.uint8)
                        
                        # Convert source and target tensors to numpy for wandb
                        src_np = sample['source_img'].permute(1, 2, 0).cpu().numpy()
                        src_np = (src_np * 255).astype(np.uint8)
                        
                        tgt_np = sample['target_img'].permute(1, 2, 0).cpu().numpy()
                        tgt_np = (tgt_np * 255).astype(np.uint8)
                        
                        # Create log data for this sample
                        sample_data = {
                            f"val_sample_{i+1}/source": wandb.Image(src_np, caption=f"Source"),
                            f"val_sample_{i+1}/target": wandb.Image(tgt_np, caption=f"Target"),
                            f"val_sample_{i+1}/generated": wandb.Image(gen_np, caption=f"Generated"),
                            f"val_sample_{i+1}/prompt": sample['prompt']
                        }
                        
                        # Add to log data
                        log_data.update(sample_data)
                        successful_images += 1
                        self.logger.info(f"✅ LoRAWandbVizHook: Generated image for sample {i+1}")
                    else:
                        self.logger.error(f"❌ LoRAWandbVizHook: No valid output image found")
                        continue
                    
                except Exception as e:
                    self.logger.error(f"❌ LoRAWandbVizHook: Error processing sample: {str(e)}")
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

    def _load_validation_data(self, solver):
        """
        Load validation data from CSV file using the path from the config
        Format: Source:FILE, Target:FILE, Prompt
        """
        # Use the paths from the solver's data config
        if self.csv_path is None:
            # Try to get paths from solver config
            try:
                if hasattr(solver, 'cfg') and solver.cfg.get('DATA') and solver.cfg.DATA.get('VAL_DATA'):
                    val_cfg = solver.cfg.DATA.VAL_DATA
                    self.csv_path = val_cfg.get('CSV_PATH', './cache/datasets/therapy_pair/images_therapist/validation.csv')
                    self.image_root_dir = val_cfg.get('IMAGE_ROOT_DIR', './cache/datasets/therapy_pair')
                    self.logger.info(f"📊 LoRAWandbVizHook: Using validation CSV: {self.csv_path}")
                    self.logger.info(f"📊 LoRAWandbVizHook: Using image root dir: {self.image_root_dir}")
                else:
                    # Default to paths from YAML file
                    self.csv_path = './cache/datasets/therapy_pair/images_therapist/validation.csv'
                    self.image_root_dir = './cache/datasets/therapy_pair'
                    self.logger.info(f"📊 LoRAWandbVizHook: Using default validation CSV: {self.csv_path}")
            except Exception as e:
                self.logger.error(f"❌ LoRAWandbVizHook: Error accessing config: {str(e)}")
                # Hardcode the path from the YAML file as fallback
                self.csv_path = './cache/datasets/therapy_pair/images_therapist/validation.csv'
                self.image_root_dir = './cache/datasets/therapy_pair'
        
        if not os.path.exists(self.csv_path):
            self.logger.error(f"❌ LoRAWandbVizHook: CSV file not found: {self.csv_path}")
            return []
        
        val_data = []
        try:
            # CSV is tab-delimited based on the sample
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    # Get source and target paths
                    source_path = os.path.join(self.image_root_dir, row['Source:FILE'])
                    target_path = os.path.join(self.image_root_dir, row['Target:FILE'])
                    prompt = row['Prompt']
                    
                    self.logger.info(f"📊 LoRAWandbVizHook: Loading source: {source_path}")
                    self.logger.info(f"📊 LoRAWandbVizHook: Loading target: {target_path}")
                    
                    try:
                        # Load source and target images
                        source_img = Image.open(source_path).convert('RGB')
                        target_img = Image.open(target_path).convert('RGB')
                        
                        # Convert to tensors
                        source_tensor = torch.from_numpy(np.array(source_img).transpose(2, 0, 1)) / 255.0
                        target_tensor = torch.from_numpy(np.array(target_img).transpose(2, 0, 1)) / 255.0
                        
                        # Create a blank mask (all ones)
                        mask_tensor = torch.ones(1, source_tensor.shape[1], source_tensor.shape[2])
                        
                        # Make sure dimensions match what the model expects
                        if source_tensor.shape[1] != source_tensor.shape[2]:
                            self.logger.warning(f"⚠️ LoRAWandbVizHook: Image dimensions not square: {source_tensor.shape}")
                        
                        val_data.append({
                            'prompt': prompt,
                            'source_img': source_tensor,
                            'target_img': target_tensor,
                            'image': source_tensor,  # Use source as input image
                            'mask': mask_tensor  # Full mask (no masking)
                        })
                    except Exception as e:
                        self.logger.error(f"❌ LoRAWandbVizHook: Error loading images: {str(e)}")
                        continue
        except Exception as e:
            self.logger.error(f"❌ LoRAWandbVizHook: Error reading CSV: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"📊 LoRAWandbVizHook: Loaded {len(val_data)} validation samples")
        return val_data

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
