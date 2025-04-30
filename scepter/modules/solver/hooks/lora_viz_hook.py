# -*- coding: utf-8 -*-
# Copyright (c) 2023

import copy
import os
import logging
import random
import traceback
import csv
import pandas as pd
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
                    
                    # Run inference using solver.run_step_test without any special parameters
                    with torch.no_grad():
                        try:
                            # Call run_step_test with simple batch list
                            results = solver.run_step_test(batch_list)
                            self.logger.info(f"✅ LoRAWandbVizHook: Results type: {type(results)}, length: {len(results) if isinstance(results, list) else 'N/A'}")
                        except Exception as e:
                            self.logger.error(f"❌ LoRAWandbVizHook: Run step test failed: {str(e)}")
                            self.logger.error(traceback.format_exc())
                            continue
                    
                    # Extract output from the results
                    output_image = None
                    
                    if isinstance(results, list) and len(results) > 0:
                        # Inspect the first result which should contain our generated image
                        result = results[0]
                        
                        # Log all available keys for debugging
                        if isinstance(result, dict):
                            self.logger.info(f"✅ LoRAWandbVizHook: Result keys: {list(result.keys())}")
                            
                            # Common keys where the output image might be found
                            for key in ['image', 'images', 'samples', 'pred', 'output']:
                                if key in result:
                                    value = result[key]
                                    self.logger.info(f"✅ LoRAWandbVizHook: Found key '{key}' with type: {type(value)}")
                                    
                                    # Handle different formats
                                    if isinstance(value, torch.Tensor):
                                        if len(value.shape) == 4:  # [batch, channels, height, width]
                                            output_image = value[0]  # Take first image
                                        elif len(value.shape) == 3:  # [channels, height, width]
                                            output_image = value
                                        break
                                    elif isinstance(value, list) and len(value) > 0:
                                        if isinstance(value[0], torch.Tensor):
                                            output_image = value[0]
                                            break
                    
                    # Check if we have a valid output image
                    if output_image is not None:
                        # Get the shape for debugging
                        self.logger.info(f"✅ LoRAWandbVizHook: Output image shape: {output_image.shape}")
                        
                        # Make sure it's the right format (C,H,W)
                        if len(output_image.shape) != 3 or output_image.shape[0] not in [1, 3, 4]:
                            self.logger.error(f"❌ LoRAWandbVizHook: Unexpected image shape: {output_image.shape}")
                            continue
                        
                        # Handle normalization - ensure it's in range [0,1]
                        if output_image.max() > 1.5:  # Likely [-1,1] or [0,255]
                            if output_image.min() < 0:
                                self.logger.info("✅ LoRAWandbVizHook: Normalizing from [-1,1] to [0,1]")
                                output_image = (output_image + 1) / 2.0
                            else:
                                self.logger.info("✅ LoRAWandbVizHook: Normalizing from [0,255] to [0,1]")
                                output_image = output_image / 255.0
                        
                        # Convert to numpy for logging
                        gen_np = output_image.permute(1, 2, 0).cpu().numpy()
                        
                        # Ensure we have values in [0,255] range for uint8
                        gen_np = (gen_np * 255).clip(0, 255).astype(np.uint8)
                        
                        # If single channel, convert to RGB
                        if gen_np.shape[2] == 1:
                            gen_np = np.repeat(gen_np, 3, axis=2)
                        
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
        Load validation data from CSV file using the path from the config.
        Uses pandas, matching the approach in CSVInRAMDataset.
        """
        # Use the paths from the solver's data config
        if self.csv_path is None:
            # Try to get paths from solver config
            try:
                # Log the solver config structure for debugging
                self.logger.info(f"📊 LoRAWandbVizHook: Solver config: {dir(solver)}")
                if hasattr(solver, 'cfg'):
                    self.logger.info(f"📊 LoRAWandbVizHook: Solver cfg: {dir(solver.cfg)}")
                
                # Try several potential paths to find the validation data config
                if hasattr(solver, 'cfg'):
                    # The YAML shows VAL_DATA at top level, so try that first
                    if hasattr(solver.cfg, 'VAL_DATA'):
                        val_cfg = solver.cfg.VAL_DATA
                        self.csv_path = val_cfg.get('CSV_PATH')
                        self.image_root_dir = val_cfg.get('IMAGE_ROOT_DIR')
                        self.logger.info(f"📊 LoRAWandbVizHook: Found config at solver.cfg.VAL_DATA")
                    # Also try DATA.VAL_DATA as seen in some other parts of the code
                    elif hasattr(solver.cfg, 'DATA') and hasattr(solver.cfg.DATA, 'VAL_DATA'):
                        val_cfg = solver.cfg.DATA.VAL_DATA
                        self.csv_path = val_cfg.get('CSV_PATH')
                        self.image_root_dir = val_cfg.get('IMAGE_ROOT_DIR')
                        self.logger.info(f"📊 LoRAWandbVizHook: Found config at solver.cfg.DATA.VAL_DATA")
                    # Try dictionary-style access as a fallback
                    elif 'VAL_DATA' in solver.cfg:
                        val_cfg = solver.cfg['VAL_DATA']
                        self.csv_path = val_cfg.get('CSV_PATH')
                        self.image_root_dir = val_cfg.get('IMAGE_ROOT_DIR')
                        self.logger.info(f"📊 LoRAWandbVizHook: Found config at solver.cfg['VAL_DATA']")
                    elif 'DATA' in solver.cfg and 'VAL_DATA' in solver.cfg['DATA']:
                        val_cfg = solver.cfg['DATA']['VAL_DATA']
                        self.csv_path = val_cfg.get('CSV_PATH')
                        self.image_root_dir = val_cfg.get('IMAGE_ROOT_DIR')
                        self.logger.info(f"📊 LoRAWandbVizHook: Found config at solver.cfg['DATA']['VAL_DATA']")
                
                # If we found a path, log it
                if self.csv_path:
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
        
        val_data = []
        try:
            # Check if file exists
            if not os.path.exists(self.csv_path):
                self.logger.error(f"❌ LoRAWandbVizHook: CSV file not found: {self.csv_path}")
                # Try to find the file in common locations
                possible_paths = [
                    os.path.join(os.getcwd(), 'validation.csv'),
                    os.path.join(os.getcwd(), 'data', 'validation.csv'),
                    os.path.join(os.getcwd(), 'datasets', 'validation.csv'),
                    './cache/datasets/therapy_pair/images_therapist/validation.csv',
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.csv_path = path
                        self.logger.info(f"📊 LoRAWandbVizHook: Found CSV at: {path}")
                        break
                
                # If still not found, use hardcoded samples
                if not os.path.exists(self.csv_path):
                    self.logger.warning(f"⚠️ LoRAWandbVizHook: Falling back to hardcoded samples")
                    return self._create_hardcoded_samples()
            
            # Try different loading approaches
            try:
                # First try pandas
                self.logger.info(f"📊 LoRAWandbVizHook: Reading CSV with pandas: {self.csv_path}")
                df = pd.read_csv(self.csv_path)
            except Exception as e:
                self.logger.error(f"❌ LoRAWandbVizHook: Error reading CSV with pandas: {str(e)}")
                self.logger.error(traceback.format_exc())
                return self._create_hardcoded_samples()
            
            # Log the columns for debugging
            self.logger.info(f"📊 LoRAWandbVizHook: CSV columns: {df.columns.tolist()}")
            
            # Check for required columns
            required_columns = ['Source:FILE', 'Prompt']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"❌ LoRAWandbVizHook: Missing required columns in CSV: {missing_columns}")
                return self._create_hardcoded_samples()
                
            # Target column is optional
            target_col = 'Target:FILE' if 'Target:FILE' in df.columns else None
            
            # Process each row
            for _, row in df.iterrows():
                source_path = os.path.join(self.image_root_dir, row['Source:FILE'])
                prompt = row['Prompt']
                
                # Target path if available
                target_path = None
                if target_col:
                    target_path = os.path.join(self.image_root_dir, row[target_col])
                
                self.logger.info(f"📊 LoRAWandbVizHook: Loading source: {source_path}")
                if target_path:
                    self.logger.info(f"📊 LoRAWandbVizHook: Loading target: {target_path}")
                
                try:
                    # Load source image
                    source_img = Image.open(source_path).convert('RGB')
                    source_tensor = torch.from_numpy(np.array(source_img).transpose(2, 0, 1)) / 255.0
                    
                    # Load target image if available
                    target_tensor = None
                    if target_path and os.path.exists(target_path):
                        target_img = Image.open(target_path).convert('RGB')
                        target_tensor = torch.from_numpy(np.array(target_img).transpose(2, 0, 1)) / 255.0
                    else:
                        # Use source as target if no target is available
                        target_tensor = source_tensor.clone()
                    
                    # Create a blank mask (all ones)
                    mask_tensor = torch.ones(1, source_tensor.shape[1], source_tensor.shape[2])
                    
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
            return self._create_hardcoded_samples()
        
        self.logger.info(f"📊 LoRAWandbVizHook: Loaded {len(val_data)} validation samples")
        
        # If no data was loaded, fall back to hardcoded samples
        if len(val_data) == 0:
            self.logger.warning(f"⚠️ LoRAWandbVizHook: No validation samples loaded, using hardcoded samples")
            return self._create_hardcoded_samples()
        
        return val_data
    
    def _create_hardcoded_samples(self):
        """
        Create hardcoded samples as a fallback when CSV loading fails
        """
        self.logger.info(f"📊 LoRAWandbVizHook: Creating hardcoded samples")
        
        # Create blank tensors of different sizes for variety
        hardcoded_samples = []
        prompts = [
            "Draw a big house with a pointy roof in a scribble style",
            "Draw a simple house with a doodle",
            "Draw a scribble of a mountain with a long slope"
        ]
        
        for i, prompt in enumerate(prompts):
            # Create a blank tensor
            size = 512
            blank_tensor = torch.zeros(3, size, size)
            mask_tensor = torch.ones(1, size, size)
            
            hardcoded_samples.append({
                'prompt': prompt,
                'source_img': blank_tensor.clone(),
                'target_img': blank_tensor.clone(),
                'image': blank_tensor,
                'mask': mask_tensor
            })
        
        self.logger.info(f"📊 LoRAWandbVizHook: Created {len(hardcoded_samples)} hardcoded samples")
        return hardcoded_samples

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
