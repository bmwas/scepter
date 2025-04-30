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
        Generate images using the current LoRA weights.
        """
        try:
            # Ensure model is in eval mode
            was_training = solver.model.training
            solver.model.eval()
            self.logger.info(f"🔄 LoRAWandbVizHook: Set solver to eval mode (was training: {was_training})")

            # Load validation samples from VAL_DATA config
            self.val_samples = self._load_validation_data(solver)
            samples_to_use = self.val_samples[:self.num_val_samples]
            num_prompts = len(samples_to_use)
            self.logger.info(f"📝 LoRAWandbVizHook: Using {num_prompts} validation samples for visualization.")

            log_data = {}
            successful_images = 0

            for i, sample in enumerate(samples_to_use):
                # Extract sample data and move to device
                prompt_text = sample['prompt']
                src = sample['source_img'].cuda(non_blocking=True)
                img = sample['image'].cuda(non_blocking=True)
                msk = sample['mask'].cuda(non_blocking=True)
                self.logger.info(f"🖼️ LoRAWandbVizHook: Processing sample {i+1}/{num_prompts}")
                self.logger.info(f"📝 LoRAWandbVizHook: Using prompt: '{prompt_text}'")
                
                output = None
                # Try direct model call first (simplest)
                try:
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                        self.logger.info(f"✅ LoRAWandbVizHook: Calling model with real sample: prompt=[[{prompt_text}]], src_image_list=[[Tensor]], src_mask_list=[[Tensor]], image=[Tensor], image_mask=[Tensor]")
                        output = solver.model(
                            prompt=[[prompt_text]],
                            src_image_list=[[src]],
                            src_mask_list=[[msk]],
                            image=[img],
                            image_mask=[msk]
                        )
                        
                        # Check if output contains an image tensor
                        if hasattr(output, 'images') and output.images is not None and len(output.images) > 0:
                            self.logger.info(f"✅ LoRAWandbVizHook: Direct call successful, image shape: {output.images[0].shape}")
                        else:
                            self.logger.warning(f"⚠️ LoRAWandbVizHook: Direct call returned output, but no images found or image list empty. Output type: {type(output)}")
                            output = None # Treat as failure if no image
                except Exception as direct_e:
                    self.logger.error(f"❌ LoRAWandbVizHook: Error with direct model call: {direct_e}")
                    self.logger.error(traceback.format_exc())
                    output = None # Ensure output is None on error

                # If direct call failed, try fallback with run_step_test using real data
                if output is None:
                    try:
                        self.logger.info(f"📝 LoRAWandbVizHook: Trying fallback with run_step_test")
                        batch_data = {
                            'prompt': [[prompt_text]],
                            'src_image_list': [[src]],
                            'src_mask_list': [[msk]],
                            'image': [img],
                            'image_mask': [msk]
                        }
                        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = solver.run_step_test(batch_data)
                        self.logger.info(f"✅ LoRAWandbVizHook: Got output from run_step_test, type: {type(outputs)}")

                        # Attempt to extract an image tensor from various possible structures
                        generated_img = None
                        if isinstance(outputs, dict):
                            # Legacy path if outputs is dict with images key
                            if 'images' in outputs and outputs['images'] is not None and len(outputs['images']) > 0:
                                generated_img = outputs['images'][0]
                            else:
                                # look for any tensor value that looks like image
                                for v in outputs.values():
                                    if isinstance(v, torch.Tensor) and v.dim() == 3:
                                        generated_img = v
                                        break
                        elif isinstance(outputs, list) and len(outputs) > 0:
                            first = outputs[0]
                            if isinstance(first, dict):
                                # Common keys
                                for k in ['reconstruct_imae', 'reconstruct_image', 'target_image', 'image']:
                                    if k in first and isinstance(first[k], torch.Tensor):
                                        generated_img = first[k]
                                        break
                                if generated_img is None:
                                    # Search any tensor
                                    for v in first.values():
                                        if isinstance(v, torch.Tensor) and v.dim() == 3:
                                            generated_img = v
                                            break

                        if generated_img is not None:
                            self.logger.info(f"✅ Extracted image tensor from outputs with shape: {generated_img.shape}")
                            # Convert to numpy
                            gen_np = generated_img.permute(1, 2, 0).detach().cpu().numpy()
                            gen_np = (gen_np * 255).clip(0, 255).astype(np.uint8)

                            # Log the image
                            sample_data = {
                                f"lora_sample_{i+1}/generated": wandb.Image(gen_np, caption="Generated"),
                                f"lora_sample_{i+1}/prompt": prompt_text
                            }
                            log_data.update(sample_data)
                            successful_images += 1
                        else:
                            self.logger.warning(f"⚠️ LoRAWandbVizHook: Could not find image tensor in outputs")
                    except Exception as fallback_e:
                        self.logger.error(f"❌ LoRAWandbVizHook: Fallback failed specific error: {fallback_e}")
                        self.logger.error(traceback.format_exc()) # Log the full traceback for the fallback
                
                # If direct call was successful, process its output
                elif hasattr(output, 'images') and output.images is not None and len(output.images) > 0:
                    generated_img = output.images[0] # First image from direct call
                    self.logger.info(f"✅ Processing image from direct call, shape: {generated_img.shape}")

                    # Convert to numpy for wandb
                    gen_np = generated_img.permute(1, 2, 0).cpu().numpy()
                    gen_np = (gen_np * 255).clip(0, 255).astype(np.uint8)

                    # Log the image
                    sample_data = {
                        f"lora_sample_{i+1}/generated": wandb.Image(gen_np, caption=f"Generated"),
                        f"lora_sample_{i+1}/prompt": prompt_text
                    }
                    log_data.update(sample_data)
                    successful_images += 1
                else:
                     self.logger.warning(f"⚠️ LoRAWandbVizHook: Direct call seemed successful but image processing failed. Output: {output}")

            # Log all collected images to wandb
            if successful_images > 0:
                wandb.log(log_data)
                self.logger.info(f"✅ LoRAWandbVizHook: Logged {successful_images}/{num_prompts} images to wandb at step {solver.iter}")
                # Restore original mode outside the loop after all prompts are processed
                if was_training:
                    solver.model.train()
                    self.logger.info(f"🔄 LoRAWandbVizHook: Restored solver to train mode")
                return True # Indicate success
            else:
                self.logger.error(f"❌ LoRAWandbVizHook: No successful images to log")
                # Restore original mode even if no images were logged
                if was_training:
                    solver.model.train()
                    self.logger.info(f"🔄 LoRAWandbVizHook: Restored solver to train mode (after failure)")
                return False # Indicate failure
                
        except Exception as e:
            self.logger.error(f"❌ LoRAWandbVizHook: Critical error in _visualize_current_lora: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Ensure model is back in train mode if it was before the main try block failed
            if 'was_training' in locals() and was_training:
                 solver.model.train()
                 self.logger.info(f"🔄 LoRAWandbVizHook: Restored solver to train mode (after critical error)")
            return False # Indicate failure

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
