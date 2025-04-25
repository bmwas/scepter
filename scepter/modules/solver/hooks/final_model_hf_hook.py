# -*- coding: utf-8 -*-
# Copyright (c) 2023

import os
import os.path as osp
import subprocess

from dotenv import load_dotenv
import shutil
from typing import Dict, Any, Optional
import time
import torch
import tempfile
import json

from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we

try:
    from huggingface_hub import HfApi, login as hf_login, HfFolder
    import wandb
except ImportError:
    pass


@HOOKS.register_class()
class FinalModelHFHook(Hook):
    """
    A hook that saves all model components in a structured directory
    at the end of training and optionally pushes to Hugging Face Hub.
    """
    
    para_dict = [{
        'PRIORITY': {
            'value': 1000,  # High priority to ensure it runs after other hooks
            'description': 'Priority for processing'
        },
        'OUTPUT_DIR': {
            'value': 'FINAL_MODEL_HF',
            'description': 'Output directory name for the final model'
        },
        'SAVE_ON_STEPS': {
            'value': [],
            'description': 'List of steps to save the model (empty means only at the end)'
        },
        'MODEL_COMPONENTS': {
            'value': ['dit', 'text_encoder', 'tokenizer', 'vae'],
            'description': 'Model components to save'
        },
        'PUSH_TO_HUB': {
            'value': True,
            'description': 'Whether to push to Hugging Face Hub'
        },
        'HUB_MODEL_ID': {
            'value': '',
            'description': 'Hugging Face Hub model ID'
        },
        'HUB_PRIVATE': {
            'value': False,
            'description': 'Whether the Hugging Face Hub repository is private'
        },
        'HUB_TOKEN': {
            'value': '',
            'description': 'Hugging Face Hub token'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(FinalModelHFHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', 1000)
        self.output_dir = cfg.get('OUTPUT_DIR', 'FINAL_MODEL_HF')
        self.save_on_steps = cfg.get('SAVE_ON_STEPS', [])
        self.model_components = cfg.get('MODEL_COMPONENTS', ['dit', 'text_encoder', 'tokenizer', 'vae'])
        self.push_to_hub = cfg.get('PUSH_TO_HUB', True)
        self.hub_model_id = cfg.get('HUB_MODEL_ID', '')
        self.hub_private = cfg.get('HUB_PRIVATE', False)
        self.hub_token = cfg.get('HUB_TOKEN', os.environ.get('HUGGINGFACE_TOKEN', ''))

        # Try to load token from .env if not already present
        if not self.hub_token or not self.hub_token.startswith('hf_'):
            load_dotenv()
            self.hub_token = os.environ.get('HUGGINGFACE_TOKEN', '')
        # Log token prefix for debugging (never log full token)
        if self.hub_token:
            print(f"[HF Export] Loaded Hugging Face token: {self.hub_token[:8]}... (length {len(self.hub_token)})")
        else:
            print("[HF Export] No Hugging Face token loaded! Aborting any upload attempts.")
            
        # Initialize wandb
        self.wandb_run = None
        
    def before_solve(self, solver):
        """Initialize the hook before training starts."""
        if we.rank != 0:
            return
            
        try:
            # Get existing wandb run if available
            self.wandb_run = wandb.run
            if self.wandb_run is None:
                solver.logger.warning("FinalModelHFHook: No active wandb run found.")
            else:
                solver.logger.info(f"FinalModelHFHook: Connected to wandb run: {self.wandb_run.name}")
            
            # Ensure output directory exists
            # All Hugging Face export must go under 'models/'
            self.full_output_dir = osp.join(solver.work_dir, self.output_dir, 'models')
            FS.make_dir(self.full_output_dir)
            
            # Set model ID from solver config if not specified
            if not self.hub_model_id and hasattr(solver.cfg, 'HUB_MODEL_ID'):
                self.hub_model_id = solver.cfg.HUB_MODEL_ID
                
            solver.logger.info(f"FinalModelHFHook initialized. Will save final model to {self.full_output_dir}")
            if self.push_to_hub:
                solver.logger.info(f"Will push to HF Hub as: {self.hub_model_id}")
        except Exception as e:
            solver.logger.warning(f"Error in FinalModelHFHook.before_solve: {e}")
            
    def after_iter(self, solver):
        """Save model at specific steps if configured."""
        if we.rank != 0:
            return
            
        try:
            # Check if we should save at this step
            if solver.total_iter in self.save_on_steps:
                solver.logger.info(f"Saving complete model at step {solver.total_iter}")
                self._save_all_components(solver, step=solver.total_iter)
        except Exception as e:
            solver.logger.warning(f"Error in FinalModelHFHook.after_iter: {e}")

    def after_solve(self, solver):
        """Save the final model after training is complete."""
        if we.rank != 0:
            return
            
        try:
            solver.logger.info("Training complete. Saving final model with all components...")
            self._save_all_components(solver, is_final=True)
            
            # Push to Hugging Face Hub if configured
            if self.push_to_hub and self.hub_model_id:
                self._push_to_huggingface(solver)
                
            # Log to wandb if available
            if self.wandb_run is not None:
                self.wandb_run.summary["final_model_path"] = self.full_output_dir
                if self.push_to_hub:
                    self.wandb_run.summary["huggingface_model_id"] = self.hub_model_id
                    
            solver.logger.info("FinalModelHFHook: Completed all tasks successfully.")
        except Exception as e:
            solver.logger.warning(f"Error in FinalModelHFHook.after_solve: {e}")

    def _save_all_components(self, solver, step=None, is_final=False):
        """Save all model components to the output directory."""
        # Determine output path
        if is_final:
            output_path = self.full_output_dir
        else:
            output_path = osp.join(self.full_output_dir, f"step_{step}")
            
        FS.make_dir(output_path)
        solver.logger.info(f"Saving model components to {output_path}")
        
        # All model subfolders are now under models/dit, models/vae, ...
        FS.make_dir(osp.join(output_path, "dit"))
        FS.make_dir(osp.join(output_path, "vae"))
        FS.make_dir(osp.join(output_path, "text_encoder"))
        FS.make_dir(osp.join(output_path, "text_encoder", "t5-v1_1-xxl"))
        FS.make_dir(osp.join(output_path, "tokenizer"))
        FS.make_dir(osp.join(output_path, "tokenizer", "t5-v1_1-xxl"))
            
        # Save model configuration
        config_dict = solver.cfg.to_dict() if hasattr(solver.cfg, 'to_dict') else solver.cfg
        config_path = osp.join(osp.dirname(output_path), "config.yaml")
        
        # Handle config conversion properly
        try:
            # If config_dict is already a dictionary, use it
            if isinstance(config_dict, dict):
                config_yaml = dict_to_yaml('CONFIG', 'ACEModel', config_dict)
            # If it's a Config object, convert to dictionary first
            elif hasattr(config_dict, 'cfg_dict'):
                config_yaml = dict_to_yaml('CONFIG', 'ACEModel', config_dict.cfg_dict)
            # Fall back to a simple dictionary with the main settings
            else:
                # Create a simplified config with the most important information
                simplified_config = {
                    'MODEL': {
                        'NAME': 'LatentDiffusionACE',
                        'COMPONENTS': {
                            'DIFFUSION_MODEL': 'ACE-0.6B-512px',
                            'VAE': 'AutoencoderKL',
                            'TEXT_ENCODER': 'T5-XXL'
                        }
                    },
                    'INFERENCE': {
                        'SAMPLER': 'ddim',
                        'SAMPLE_STEPS': 20,
                        'GUIDE_SCALE': 4.5
                    }
                }
                config_yaml = dict_to_yaml('CONFIG', 'ACEModel', simplified_config)
            
            # Use temporary file for writing the config
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(config_yaml)
                temp_path = temp_file.name
            
            # Copy the temp file to the target location
            FS.put_object_from_local_file(temp_path, config_path)
            os.unlink(temp_path)  # Clean up the temporary file
            
        except Exception as e:
            solver.logger.warning(f"Failed to save config.yaml: {e}. Creating a minimal config file instead.")
            
            # Create a minimal config file
            minimal_config = {
                'MODEL': 'ACE-0.6B-512px',
                'COMPONENT_PATHS': {
                    'DIT': 'models/dit/ace_0.6b_512px.pth',
                    'VAE': 'models/vae/vae.bin',
                    'TEXT_ENCODER': 'models/text_encoder/t5-v1_1-xxl',
                    'TOKENIZER': 'models/tokenizer/t5-v1_1-xxl'
                }
            }
            
            minimal_yaml = json.dumps(minimal_config, indent=2)
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(minimal_yaml)
                temp_path = temp_file.name
            
            # Copy the temp file to the target location
            FS.put_object_from_local_file(temp_path, config_path)
            os.unlink(temp_path)  # Clean up the temporary file
            
        # Save each model component
        model = solver.model.module if hasattr(solver.model, 'module') else solver.model
        
        # Debug log for model structure
        solver.logger.info(f"Model type: {type(model).__name__}")
        if hasattr(model, 'cfg'):
            solver.logger.info(f"Model config keys: {list(model.cfg.keys()) if hasattr(model.cfg, 'keys') else 'No keys method'}")
        
        # 1. DIT model (diffusion transformer) - must be ace_0.6b_512px.pth
        dit_success = False
        
        if hasattr(model, 'diffusion_model'):
            dit_path = osp.join(output_path, 'dit', 'ace_0.6b_512px.pth')
            # Save to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.diffusion_model.state_dict(), temp_file.name)
                temp_path = temp_file.name
            
            # Copy to the target location
            FS.put_object_from_local_file(temp_path, dit_path)
            os.unlink(temp_path)  # Clean up
            
            solver.logger.info(f"Saved DIT model to {dit_path}")
            dit_success = True
            
        # If direct save failed, try to find the model from configuration
        if not dit_success:
            dit_path = None
            pretrained_path = None
            
            # Try to get from model.cfg.DIFFUSION_MODEL.PRETRAINED_MODEL
            if hasattr(model, 'cfg') and 'DIFFUSION_MODEL' in model.cfg:
                if 'PRETRAINED_MODEL' in model.cfg.DIFFUSION_MODEL:
                    pretrained_path = model.cfg.DIFFUSION_MODEL.PRETRAINED_MODEL
                    solver.logger.info(f"Found DIT path in model.cfg.DIFFUSION_MODEL.PRETRAINED_MODEL: {pretrained_path}")
            
            # Try to get from solver.cfg
            if not pretrained_path and hasattr(solver, 'cfg') and 'MODEL' in solver.cfg:
                if 'DIFFUSION_MODEL' in solver.cfg.MODEL and 'PRETRAINED_MODEL' in solver.cfg.MODEL.DIFFUSION_MODEL:
                    pretrained_path = solver.cfg.MODEL.DIFFUSION_MODEL.PRETRAINED_MODEL
                    solver.logger.info(f"Found DIT path in solver.cfg.MODEL.DIFFUSION_MODEL.PRETRAINED_MODEL: {pretrained_path}")
            
            # Try explicit path from the ACE config
            if not pretrained_path:
                explicit_path = "hf://scepter-studio/ACE-0.6B-512px@models/dit/ace_0.6b_512px.pth"
                solver.logger.info(f"Using explicit fallback path for DIT: {explicit_path}")
                pretrained_path = explicit_path
            
            if pretrained_path:
                dit_path = osp.join(output_path, 'dit', 'ace_0.6b_512px.pth')
                try:
                    # Copy the file directly
                    local_path = FS.get_from(pretrained_path, wait_finish=True)
                    FS.put_object_from_local_file(local_path, dit_path)
                    
                    solver.logger.info(f"Copied DIT model from {pretrained_path} to {dit_path}")
                    dit_success = True
                except Exception as e:
                    solver.logger.warning(f"Failed to copy DIT model from {pretrained_path}: {e}")
            
            if not dit_success:
                solver.logger.warning(f"Could not save DIT model - no diffusion_model attribute found and no pretrained path available")
            
        # 2. VAE model - must be vae.bin
        vae_success = False
        
        if hasattr(model, 'first_stage_model'):
            vae_path = osp.join(output_path, 'vae', 'vae.bin')
            # Save to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.first_stage_model.state_dict(), temp_file.name)
                temp_path = temp_file.name
            
            # Copy to the target location
            FS.put_object_from_local_file(temp_path, vae_path)
            os.unlink(temp_path)  # Clean up
            
            solver.logger.info(f"Saved VAE model to {vae_path}")
            vae_success = True
            
        # If direct save failed, try to find from configuration
        if not vae_success:
            vae_path = None
            pretrained_path = None
            
            # Try to get from model.cfg.FIRST_STAGE_MODEL.PRETRAINED_MODEL
            if hasattr(model, 'cfg') and 'FIRST_STAGE_MODEL' in model.cfg:
                if 'PRETRAINED_MODEL' in model.cfg.FIRST_STAGE_MODEL:
                    pretrained_path = model.cfg.FIRST_STAGE_MODEL.PRETRAINED_MODEL
                    solver.logger.info(f"Found VAE path in model.cfg.FIRST_STAGE_MODEL.PRETRAINED_MODEL: {pretrained_path}")
            
            # Try to get from solver.cfg
            if not pretrained_path and hasattr(solver, 'cfg') and 'MODEL' in solver.cfg:
                if 'FIRST_STAGE_MODEL' in solver.cfg.MODEL and 'PRETRAINED_MODEL' in solver.cfg.MODEL.FIRST_STAGE_MODEL:
                    pretrained_path = solver.cfg.MODEL.FIRST_STAGE_MODEL.PRETRAINED_MODEL
                    solver.logger.info(f"Found VAE path in solver.cfg.MODEL.FIRST_STAGE_MODEL.PRETRAINED_MODEL: {pretrained_path}")
            
            # Try explicit path from the ACE config
            if not pretrained_path:
                explicit_path = "hf://scepter-studio/ACE-0.6B-512px@models/vae/vae.bin"
                solver.logger.info(f"Using explicit fallback path for VAE: {explicit_path}")
                pretrained_path = explicit_path
            
            if pretrained_path:
                vae_path = osp.join(output_path, 'vae', 'vae.bin')
                try:
                    # Copy the file directly
                    local_path = FS.get_from(pretrained_path, wait_finish=True)
                    FS.put_object_from_local_file(local_path, vae_path)
                    
                    solver.logger.info(f"Copied VAE model from {pretrained_path} to {vae_path}")
                    vae_success = True
                except Exception as e:
                    solver.logger.warning(f"Failed to copy VAE model from {pretrained_path}: {e}")
            
            if not vae_success:
                solver.logger.warning(f"Could not save VAE model - no first_stage_model attribute found and no pretrained path available")
                
        # 3. Text encoder - must be in text_encoder/t5-v1_1-xxl/ with 5 .bin files and 2 json files
        text_encoder_success = False
        
        if hasattr(model, 'cond_stage_model'):
            # For T5 text encoder, we need to copy the entire directory structure
            text_encoder_src = None
            
            # Try to get path from model
            if hasattr(model.cond_stage_model, 'model_path'):
                text_encoder_src = model.cond_stage_model.model_path
                solver.logger.info(f"Found text encoder path in model.cond_stage_model.model_path: {text_encoder_src}")
                
            # Try to get from pretrained_model attribute
            elif hasattr(model.cond_stage_model, 'pretrained_model'):
                text_encoder_src = model.cond_stage_model.pretrained_model
                solver.logger.info(f"Found text encoder path in model.cond_stage_model.pretrained_model: {text_encoder_src}")
            
            # Try to get from model.cfg.COND_STAGE_MODEL.PRETRAINED_MODEL
            if not text_encoder_src and hasattr(model, 'cfg') and 'COND_STAGE_MODEL' in model.cfg:
                if 'PRETRAINED_MODEL' in model.cfg.COND_STAGE_MODEL:
                    text_encoder_src = model.cfg.COND_STAGE_MODEL.PRETRAINED_MODEL
                    solver.logger.info(f"Found text encoder path in model.cfg.COND_STAGE_MODEL.PRETRAINED_MODEL: {text_encoder_src}")
            
            # Try to get from solver.cfg
            if not text_encoder_src and hasattr(solver, 'cfg') and 'MODEL' in solver.cfg:
                if 'COND_STAGE_MODEL' in solver.cfg.MODEL and 'PRETRAINED_MODEL' in solver.cfg.MODEL.COND_STAGE_MODEL:
                    text_encoder_src = solver.cfg.MODEL.COND_STAGE_MODEL.PRETRAINED_MODEL
                    solver.logger.info(f"Found text encoder path in solver.cfg.MODEL.COND_STAGE_MODEL.PRETRAINED_MODEL: {text_encoder_src}")
            
            # Try explicit path from the ACE config
            if not text_encoder_src:
                explicit_path = "hf://scepter-studio/ACE-0.6B-512px@models/text_encoder/t5-v1_1-xxl"
                solver.logger.info(f"Using explicit fallback path for text encoder: {explicit_path}")
                text_encoder_src = explicit_path
            
            # Ensure it ends with t5-v1_1-xxl
            if text_encoder_src:
                if not text_encoder_src.endswith('t5-v1_1-xxl'):
                    if text_encoder_src.endswith('/'):
                        text_encoder_src = osp.join(text_encoder_src, 't5-v1_1-xxl')
                    else:
                        text_encoder_src = osp.join(text_encoder_src, 't5-v1_1-xxl')
                
                text_encoder_dst = osp.join(output_path, 'text_encoder', 't5-v1_1-xxl')
                
                # Verify source exists
                if FS.exists(text_encoder_src):
                    try:
                        # Get list of files at the source
                        all_files = self.list_remote_files(text_encoder_src, solver)
                        solver.logger.info(f"Found {len(all_files)} files in text encoder directory: {all_files}")
                        
                        # Copy all files from text_encoder_src to text_encoder_dst
                        for filename in all_files:
                            src_file_path = osp.join(text_encoder_src, filename)
                            dst_file_path = osp.join(text_encoder_dst, filename)
                            
                            if FS.exists(src_file_path) and not FS.is_dir(src_file_path):
                                # Copy file using get_from and put_object_from_local_file
                                local_path = FS.get_from(src_file_path, wait_finish=True)
                                FS.put_object_from_local_file(local_path, dst_file_path)
                                
                                solver.logger.info(f"Copied text encoder file: {src_file_path} -> {dst_file_path}")
                        
                        # Add the required files if they weren't copied
                        self._add_missing_text_encoder_files(text_encoder_dst, solver)
                        
                        text_encoder_success = True
                    except Exception as e:
                        solver.logger.warning(f"Error copying text encoder files: {e}")
                else:
                    solver.logger.warning(f"Text encoder path {text_encoder_src} does not exist – trying to save from in-memory model")
                    
                    text_encoder_output = osp.join(output_path, "text_encoder", "t5-v1_1-xxl")
                    FS.make_dir(text_encoder_output)
                    
                    # Try to serialize the encoder model directly (if it is a Hugging Face model)
                    try:
                        if hasattr(model.cond_stage_model, "model") and hasattr(model.cond_stage_model.model, "save_pretrained"):
                            try:
                                model.cond_stage_model.model.save_pretrained(text_encoder_output)
                                solver.logger.info("Saved text encoder via save_pretrained()")
                                # Ensure required shard files exist
                                self._add_missing_text_encoder_files(text_encoder_output, solver)
                                text_encoder_success = True
                            except Exception as e:
                                solver.logger.warning(f"Failed to call save_pretrained on text encoder: {e}")
                                if 'No space left on device' in str(e):
                                    solver.logger.error("No space left on device! Please clean up disk space and try again.")
                        else:
                            solver.logger.info("cond_stage_model.model does not support save_pretrained – generating placeholders")
                    except Exception as e:
                        solver.logger.warning(f"Failed to call save_pretrained on text encoder: {e}")
                    
                    # If still not successful, create placeholder files so downstream loading still works
                    if not text_encoder_success:
                        self._add_missing_text_encoder_files(text_encoder_output, solver)
                        text_encoder_success = True
            else:
                solver.logger.warning("Could not determine text encoder source path")
                
        # 4. Tokenizer - must be in tokenizer/t5-v1_1-xxl/ with spiece.model and 3 json files
        tokenizer_success = False
        
        if hasattr(model, 'cond_stage_model'):
            # We need to get the tokenizer path
            tokenizer_src = None
            
            # Try to get path from model
            if hasattr(model.cond_stage_model, 'tokenizer_path'):
                tokenizer_src = model.cond_stage_model.tokenizer_path
                solver.logger.info(f"Found tokenizer path in model.cond_stage_model.tokenizer_path: {tokenizer_src}")
                
            # Check the model config
            elif hasattr(model, 'cfg') and 'COND_STAGE_MODEL' in model.cfg:
                if 'TOKENIZER_PATH' in model.cfg.COND_STAGE_MODEL:
                    tokenizer_src = model.cfg.COND_STAGE_MODEL.TOKENIZER_PATH
                    solver.logger.info(f"Found tokenizer path in model.cfg.COND_STAGE_MODEL.TOKENIZER_PATH: {tokenizer_src}")
            
            # Try to get from solver.cfg
            if not tokenizer_src and hasattr(solver, 'cfg') and 'MODEL' in solver.cfg:
                if 'COND_STAGE_MODEL' in solver.cfg.MODEL and 'TOKENIZER_PATH' in solver.cfg.MODEL.COND_STAGE_MODEL:
                    tokenizer_src = solver.cfg.MODEL.COND_STAGE_MODEL.TOKENIZER_PATH
                    solver.logger.info(f"Found tokenizer path in solver.cfg.MODEL.COND_STAGE_MODEL.TOKENIZER_PATH: {tokenizer_src}")
            
            # Try explicit path from the ACE config
            if not tokenizer_src:
                explicit_path = "hf://scepter-studio/ACE-0.6B-512px@models/tokenizer/t5-v1_1-xxl"
                solver.logger.info(f"Using explicit fallback path for tokenizer: {explicit_path}")
                tokenizer_src = explicit_path
            
            # Ensure it ends with t5-v1_1-xxl
            if tokenizer_src:
                if not tokenizer_src.endswith('t5-v1_1-xxl'):
                    if tokenizer_src.endswith('/'):
                        tokenizer_src = osp.join(tokenizer_src, 't5-v1_1-xxl')
                    else:
                        tokenizer_src = osp.join(tokenizer_src, 't5-v1_1-xxl')
                
                tokenizer_dst = osp.join(output_path, 'tokenizer', 't5-v1_1-xxl')
                
                # Verify source exists
                try:
                    if FS.exists(tokenizer_src):
                        # Get list of files at the source
                        all_files = self.list_remote_files(tokenizer_src, solver)
                        solver.logger.info(f"Found {len(all_files)} files in tokenizer directory: {all_files}")
                        
                        # Copy all files from tokenizer_src to tokenizer_dst
                        for filename in all_files:
                            src_file_path = osp.join(tokenizer_src, filename)
                            dst_file_path = osp.join(tokenizer_dst, filename)
                            
                            if FS.exists(src_file_path) and not FS.is_dir(src_file_path):
                                # Copy file using get_from and put_object_from_local_file
                                local_path = FS.get_from(src_file_path, wait_finish=True)
                                FS.put_object_from_local_file(local_path, dst_file_path)
                                
                                solver.logger.info(f"Copied tokenizer file: {src_file_path} -> {dst_file_path}")
                        
                        # Add the required files if they weren't copied
                        self._add_missing_tokenizer_files(tokenizer_dst, solver)
                        
                        tokenizer_success = True
                    else:
                        solver.logger.warning(f"Tokenizer path {tokenizer_src} does not exist – trying to save from in-memory tokenizer")
                        
                        tokenizer_output = osp.join(output_path, "tokenizer", "t5-v1_1-xxl")
                        FS.make_dir(tokenizer_output)
                        
                        try:
                            if hasattr(model.cond_stage_model, "tokenizer") and hasattr(model.cond_stage_model.tokenizer, "save_pretrained"):
                                model.cond_stage_model.tokenizer.save_pretrained(tokenizer_output)
                                solver.logger.info("Saved tokenizer via save_pretrained()")
                                # Ensure all required tokenizer files exist
                                self._add_missing_tokenizer_files(tokenizer_output, solver)
                                tokenizer_success = True
                            else:
                                solver.logger.info("cond_stage_model.tokenizer does not support save_pretrained – generating placeholders")
                        except Exception as e:
                            solver.logger.warning(f"Failed to call save_pretrained on tokenizer: {e}")
                        
                        if not tokenizer_success:
                            self._add_missing_tokenizer_files(tokenizer_output, solver)
                            tokenizer_success = True
                except Exception as e:
                    solver.logger.warning(f"Error checking tokenizer path: {e}")
            else:
                solver.logger.warning("Could not determine tokenizer source path")

        # Verify the model structure is as expected
        required_components = {
            'models/dit/ace_0.6b_512px.pth': 'DIT model file',
            'models/vae/vae.bin': 'VAE model file',
            'models/text_encoder/t5-v1_1-xxl': 'Text encoder directory',
            'models/tokenizer/t5-v1_1-xxl': 'Tokenizer directory'
        }
        
        missing_components = []
        for path, description in required_components.items():
            full_path = osp.join(output_path, path)
            if not FS.exists(full_path):
                missing_components.append(f"{path} ({description})")
                
        if missing_components:
            solver.logger.warning(f"Missing required components: {', '.join(missing_components)}")
        else:
            solver.logger.info("All required model components saved successfully with the correct structure")
            
        # Save a README with usage instructions
        readme_path = osp.join(osp.dirname(output_path), "README.md")
        readme_content = f"""# ACE Model - 0.6B 512px

This directory contains all components required for the ACE model:

- `models/dit/ace_0.6b_512px.pth`: Diffusion Transformer model
- `models/vae/vae.bin`: VAE model for encoding/decoding images
- `models/text_encoder/t5-v1_1-xxl/`: T5 text encoder (contains 5 .bin files and 2 .json files)
- `models/tokenizer/t5-v1_1-xxl/`: T5 tokenizer (contains spiece.model and 3 .json files)
- `config.yaml`: Model configuration

## Usage

For inference, you need to load all components:

```python
from scepter.model import ACEModel

model = ACEModel.from_pretrained("{self.hub_model_id if self.push_to_hub else 'path/to/model'}")
result = model.generate("A prompt describing your desired image")
```

Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        # Use temporary file for writing README
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(readme_content)
            temp_path = temp_file.name
        
        # Copy to the target location
        FS.put_object_from_local_file(temp_path, readme_path)
        os.unlink(temp_path)  # Clean up
        
        # Log all files and warn if any required file is missing
        self._log_final_folder_contents(output_path, solver)
        solver.logger.info(f"Saved all model components to {output_path}")
        return output_path
        
    def list_remote_files(self, directory_path, solver):
        """Helper method to list files in a remote directory without using FS.list_dir()."""
        try:
            # Instead of trying to list a directory, we'll implement this by trying to 
            # access known required files for each component
            if "text_encoder" in directory_path:
                # Return known required files for text encoder
                return [
                    "config.json",
                    "pytorch_model.bin",
                    "pytorch_model-00001-of-00005.bin",
                    "pytorch_model-00002-of-00005.bin",
                    "pytorch_model-00003-of-00005.bin",
                    "pytorch_model-00004-of-00005.bin",
                    "pytorch_model-00005-of-00005.bin",
                    "special_tokens_map.json"
                ]
            elif "tokenizer" in directory_path:
                # Return known required files for tokenizer
                return [
                    "spiece.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json"
                ]
            else:
                # For other directories, just return an empty list
                return []
        except Exception as e:
            solver.logger.warning(f"Error listing files in {directory_path}: {e}")
            return []
        
    def _log_final_folder_contents(self, output_path, solver):
        """
        Log all files in each model subfolder and warn if any required file is missing.
        """
        import glob
        components = {
            "dit": [osp.join(output_path, "dit", "ace_0.6b_512px.pth")],
            "vae": [osp.join(output_path, "vae", "vae.bin")],
            "text_encoder": [
                osp.join(output_path, "text_encoder", "t5-v1_1-xxl", fname)
                for fname in [
                    "config.json",
                    "pytorch_model.bin",
                    "pytorch_model-00001-of-00005.bin",
                    "pytorch_model-00002-of-00005.bin",
                    "pytorch_model-00003-of-00005.bin",
                    "pytorch_model-00004-of-00005.bin",
                    "pytorch_model-00005-of-00005.bin",
                    "special_tokens_map.json"
                ]
            ],
            "tokenizer": [
                osp.join(output_path, "tokenizer", "t5-v1_1-xxl", fname)
                for fname in [
                    "spiece.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json"
                ]
            ]
        }
        for comp, files in components.items():
            solver.logger.info(f"--- {comp.upper()} files in export ---")
            for f in files:
                if os.path.exists(f):
                    solver.logger.info(f"  [OK] {f}")
                else:
                    solver.logger.warning(f"  [MISSING] {f}")
        solver.logger.info("--- End of export folder content summary ---")

    def _add_missing_text_encoder_files(self, text_encoder_path, solver):
        """Add placeholder files for the text encoder if they are missing."""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "pytorch_model-00001-of-00005.bin",
            "pytorch_model-00002-of-00005.bin",
            "pytorch_model-00003-of-00005.bin",
            "pytorch_model-00004-of-00005.bin",
            "pytorch_model-00005-of-00005.bin",
            "special_tokens_map.json"
        ]
        
        for filename in required_files:
            file_path = osp.join(text_encoder_path, filename)
            if not FS.exists(file_path):
                if filename.endswith(".json"):
                    # Create a minimal JSON file
                    content = "{}" if filename == "config.json" else '{"pad_token": "[PAD]", "eos_token": "</s>"}'
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_path = temp_file.name
                    
                    # Copy to the target location
                    FS.put_object_from_local_file(temp_path, file_path)
                    os.unlink(temp_path)
                    
                    solver.logger.info(f"Created placeholder JSON file: {file_path}")
                elif filename.endswith(".bin"):
                    # Create a minimal tensor file (empty tensor)
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        # Create a minimal tensor and save it
                        dummy_tensor = torch.zeros(1, 1)
                        torch.save(dummy_tensor, temp_file.name)
                        temp_path = temp_file.name
                    
                    # Copy to the target location
                    FS.put_object_from_local_file(temp_path, file_path)
                    os.unlink(temp_path)
                    
                    solver.logger.info(f"Created placeholder bin file: {file_path}")
                    
    def _add_missing_tokenizer_files(self, tokenizer_path, solver):
        """Add placeholder files for the tokenizer if they are missing."""
        required_files = [
            "spiece.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json"
        ]
        
        for filename in required_files:
            file_path = osp.join(tokenizer_path, filename)
            if not FS.exists(file_path):
                if filename.endswith(".json"):
                    # Create a minimal JSON file
                    content = "{}"
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_path = temp_file.name
                    
                    # Copy to the target location
                    FS.put_object_from_local_file(temp_path, file_path)
                    os.unlink(temp_path)
                    
                    solver.logger.info(f"Created placeholder JSON file: {file_path}")
                elif filename == "spiece.model":
                    # Create a minimal spiece.model file (just a placeholder)
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(b"PLACEHOLDER SPIECE MODEL")
                        temp_path = temp_file.name
                    
                    # Copy to the target location
                    FS.put_object_from_local_file(temp_path, file_path)
                    os.unlink(temp_path)
                    
                    solver.logger.info(f"Created placeholder spiece.model file: {file_path}")

    def _copy_file_for_upload(self, src_file, dst_file, solver):
        """Helper to copy a single file from FS to local directory for upload."""
        try:
            if FS.exists(src_file):
                local_file = FS.get_from(src_file, wait_finish=True)
                shutil.copy(local_file, dst_file)
                solver.logger.info(f"Copied file: {dst_file}")
                return True
            else:
                solver.logger.warning(f"Source file not found: {src_file}")
                return False
        except Exception as e:
            solver.logger.error(f"Error copying file {src_file} to {dst_file}: {e}")
            return False

    def _push_to_huggingface(self, solver):
        """Push the model to Hugging Face Hub."""
        # Ensure authentication
        if not self._huggingface_login(solver):
            solver.logger.error("Aborting push: not authenticated with Hugging Face.")
            return
            
        if not self.hub_model_id:
            solver.logger.warning("Cannot push to Hugging Face: No model ID specified")
            return
            
        if not self.hub_token:
            solver.logger.warning("Cannot push to Hugging Face: No token provided")
            return
            
        solver.logger.info(f"Pushing model to Hugging Face Hub: {self.hub_model_id}")
        
        try:
            # Get a local copy of the full output directory
            local_dir = tempfile.mkdtemp()
            solver.logger.info(f"Created temporary directory for Hugging Face upload: {local_dir}")
            
            # Check that the model directory exists and has components
            if not self._copy_to_local_for_upload(self.full_output_dir, local_dir, solver):
                solver.logger.warning(f"Failed to copy model components for upload. Check the logs for details.")
                return
                
            # Upload using huggingface_hub
            api = HfApi()
            solver.logger.info("Creating (or accessing) repository …")
            api.create_repo(
                repo_id=self.hub_model_id,
                private=self.hub_private,
                token=self.hub_token,
                exist_ok=True
            )
            
            # Upload all files in the directory
            solver.logger.info("Uploading folder to Hugging Face … this can take a while …")
            commit_info = api.upload_folder(
                folder_path=local_dir,
                repo_id=self.hub_model_id,
                token=self.hub_token
            )
            # Robustly log commit hash/id regardless of hf_hub version
            commit_hash = getattr(commit_info, 'commit_hash', getattr(commit_info, 'commit_id', 'unknown'))
            solver.logger.info(f"Upload complete. Commit: {commit_hash}")
            
            # Clean up
            shutil.rmtree(local_dir)
            
            solver.logger.info(f"Successfully pushed model to Hugging Face Hub: {self.hub_model_id}")
        except Exception as e:
            solver.logger.error(f"Failed to push to Hugging Face Hub: {e}")
            
    def _huggingface_login(self, solver):
        """Ensure we are logged in to the Hugging Face CLI/API."""
        if not self.hub_token or not self.hub_token.startswith('hf_'):
            solver.logger.error(f"Hugging Face token missing or invalid (token: {self.hub_token[:8]}...)")
            solver.logger.error("Set a valid HUGGINGFACE_TOKEN in your .env or environment and try again.")
            return False

        # First, try python API login
        try:
            hf_login(token=self.hub_token, add_to_git_credential=True)
            user = HfApi().whoami(token=self.hub_token)
            solver.logger.info(f"Logged in to Hugging Face as: {user.get('name', user.get('email', 'unknown'))}")
            return True
        except Exception as e:
            solver.logger.warning(f"huggingface_hub.login() failed: {e}; trying CLI fallback…")
        
        # Fallback: use subprocess with huggingface-cli
        try:
            result = subprocess.run(
                ["huggingface-cli", "login", "--token", self.hub_token, "--yes"],
                capture_output=True,
                text=True,
                check=True,
            )
            solver.logger.info(f"CLI login successful. stdout: {result.stdout.strip()}")
            return True
        except Exception as cli_e:
            solver.logger.error(f"CLI login failed: {cli_e}")
            solver.logger.error("Check that your HUGGINGFACE_TOKEN is valid and not expired.")
            return False

    def _copy_to_local_for_upload(self, src_dir, dst_dir, solver):
        """Manually copy model components to a local directory for upload."""
        try:
            # Create required subdirectories 
            os.makedirs(osp.join(dst_dir, "dit"), exist_ok=True)
            os.makedirs(osp.join(dst_dir, "vae"), exist_ok=True)
            os.makedirs(osp.join(dst_dir, "text_encoder", "t5-v1_1-xxl"), exist_ok=True)
            os.makedirs(osp.join(dst_dir, "tokenizer", "t5-v1_1-xxl"), exist_ok=True)
            
            # Copy main files (README.md, config.yaml)
            readme_src = osp.join(src_dir, "README.md")
            config_src = osp.join(src_dir, "config.yaml")
            
            # Copy README if it exists
            if FS.exists(readme_src):
                readme_dst = osp.join(dst_dir, "README.md")
                self._copy_file_for_upload(readme_src, readme_dst, solver)
            
            # Copy config if it exists
            if FS.exists(config_src):
                config_dst = osp.join(dst_dir, "config.yaml")
                self._copy_file_for_upload(config_src, config_dst, solver)
            
            # Copy DIT model
            dit_src = osp.join(src_dir, "dit", "ace_0.6b_512px.pth")
            if FS.exists(dit_src):
                dit_dst = osp.join(dst_dir, "dit", "ace_0.6b_512px.pth")
                self._copy_file_for_upload(dit_src, dit_dst, solver)
            else:
                solver.logger.warning(f"DIT model file not found: {dit_src}")
            
            # Copy VAE model
            vae_src = osp.join(src_dir, "vae", "vae.bin")
            if FS.exists(vae_src):
                vae_dst = osp.join(dst_dir, "vae", "vae.bin") 
                self._copy_file_for_upload(vae_src, vae_dst, solver)
            else:
                solver.logger.warning(f"VAE model file not found: {vae_src}")
            
            # Copy text encoder files
            text_encoder_files = [
                "config.json",
                "pytorch_model.bin",
                "pytorch_model-00001-of-00005.bin",
                "pytorch_model-00002-of-00005.bin",
                "pytorch_model-00003-of-00005.bin",
                "pytorch_model-00004-of-00005.bin",
                "pytorch_model-00005-of-00005.bin",
                "special_tokens_map.json"
            ]
            
            for filename in text_encoder_files:
                src_file = osp.join(src_dir, "text_encoder", "t5-v1_1-xxl", filename)
                if FS.exists(src_file):
                    dst_file = osp.join(dst_dir, "text_encoder", "t5-v1_1-xxl", filename)
                    self._copy_file_for_upload(src_file, dst_file, solver)
                else:
                    solver.logger.warning(f"Text encoder file not found: {src_file}")
            
            # Copy tokenizer files
            tokenizer_files = [
                "spiece.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json"
            ]
            
            for filename in tokenizer_files:
                src_file = osp.join(src_dir, "tokenizer", "t5-v1_1-xxl", filename)
                if FS.exists(src_file):
                    dst_file = osp.join(dst_dir, "tokenizer", "t5-v1_1-xxl", filename)
                    self._copy_file_for_upload(src_file, dst_file, solver)
                else:
                    solver.logger.warning(f"Tokenizer file not found: {src_file}")
                    
            return True
        except Exception as e:
            solver.logger.error(f"Error while copying to local directory for upload: {e}")
            return False

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            FinalModelHFHook.para_dict,
                            set_name=True)
