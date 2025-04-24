# -*- coding: utf-8 -*-
# Copyright (c) 2023

import os
import os.path as osp
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
    from huggingface_hub import HfApi
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
            self.full_output_dir = osp.join(solver.work_dir, self.output_dir)
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
        
        # Create subdirectories for each component with required structure
        # dit/ - contains ace_0.6b_512px.pth
        FS.make_dir(osp.join(output_path, "dit"))
        
        # vae/ - contains vae.bin
        FS.make_dir(osp.join(output_path, "vae"))
        
        # text_encoder/ - contains t5-v1_1-xxl/ subfolder with model files
        FS.make_dir(osp.join(output_path, "text_encoder"))
        FS.make_dir(osp.join(output_path, "text_encoder", "t5-v1_1-xxl"))
        
        # tokenizer/ - contains t5-v1_1-xxl/ subfolder with tokenizer files
        FS.make_dir(osp.join(output_path, "tokenizer"))
        FS.make_dir(osp.join(output_path, "tokenizer", "t5-v1_1-xxl"))
            
        # Save model configuration
        config_dict = solver.cfg.to_dict() if hasattr(solver.cfg, 'to_dict') else solver.cfg
        config_path = osp.join(output_path, "config.yaml")
        
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
                    'DIT': 'dit/ace_0.6b_512px.pth',
                    'VAE': 'vae/vae.bin',
                    'TEXT_ENCODER': 'text_encoder/t5-v1_1-xxl',
                    'TOKENIZER': 'tokenizer/t5-v1_1-xxl'
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
        
        # 1. DIT model (diffusion transformer) - must be ace_0.6b_512px.pth
        if 'dit' in self.model_components and hasattr(model, 'diffusion_model'):
            dit_path = osp.join(output_path, 'dit', 'ace_0.6b_512px.pth')
            # Save to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.diffusion_model.state_dict(), temp_file.name)
                temp_path = temp_file.name
            
            # Copy to the target location
            FS.put_object_from_local_file(temp_path, dit_path)
            os.unlink(temp_path)  # Clean up
            
            solver.logger.info(f"Saved DIT model to {dit_path}")
        else:
            # Try to find the DIT model from the original pretrained path
            if hasattr(model, 'diffusion_model') and hasattr(model.diffusion_model, 'pretrained_model'):
                pretrained_dit = model.diffusion_model.pretrained_model
                if pretrained_dit and FS.exists(pretrained_dit):
                    dit_path = osp.join(output_path, 'dit', 'ace_0.6b_512px.pth')
                    # Copy the file directly
                    local_path = FS.get_from(pretrained_dit, wait_finish=True)
                    FS.put_object_from_local_file(local_path, dit_path)
                    
                    solver.logger.info(f"Copied DIT model from {pretrained_dit} to {dit_path}")
                else:
                    solver.logger.warning(f"DIT model not found at {pretrained_dit}")
            else:
                solver.logger.warning("Could not save DIT model - no diffusion_model attribute found")
            
        # 2. VAE model - must be vae.bin
        if 'vae' in self.model_components and hasattr(model, 'first_stage_model'):
            vae_path = osp.join(output_path, 'vae', 'vae.bin')
            # Save to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.first_stage_model.state_dict(), temp_file.name)
                temp_path = temp_file.name
            
            # Copy to the target location
            FS.put_object_from_local_file(temp_path, vae_path)
            os.unlink(temp_path)  # Clean up
            
            solver.logger.info(f"Saved VAE model to {vae_path}")
        else:
            # Try to get from original pretrained path
            if hasattr(model, 'first_stage_model') and hasattr(model.first_stage_model, 'pretrained_model'):
                pretrained_vae = model.first_stage_model.pretrained_model
                if pretrained_vae and FS.exists(pretrained_vae):
                    vae_path = osp.join(output_path, 'vae', 'vae.bin')
                    # Copy the file directly
                    local_path = FS.get_from(pretrained_vae, wait_finish=True)
                    FS.put_object_from_local_file(local_path, vae_path)
                    
                    solver.logger.info(f"Copied VAE model from {pretrained_vae} to {vae_path}")
                else:
                    solver.logger.warning(f"VAE model not found at {pretrained_vae}")
            else:
                solver.logger.warning("Could not save VAE model - no first_stage_model attribute found")
                
        # 3. Text encoder - must be in text_encoder/t5-v1_1-xxl/ with 5 bin files and 2 json files
        if 'text_encoder' in self.model_components and hasattr(model, 'cond_stage_model'):
            # For T5 text encoder, we need to copy the entire directory structure
            text_encoder_src = None
            
            # Try to get path from model
            if hasattr(model.cond_stage_model, 'model_path'):
                text_encoder_src = model.cond_stage_model.model_path
            # Try to get from pretrained_model attribute
            elif hasattr(model.cond_stage_model, 'pretrained_model'):
                text_encoder_src = model.cond_stage_model.pretrained_model
            
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
                    # Copy all files from text_encoder_src to text_encoder_dst
                    file_list = FS.list_dir(text_encoder_src)
                    for filename in file_list:
                        src_file_path = osp.join(text_encoder_src, filename)
                        dst_file_path = osp.join(text_encoder_dst, filename)
                        
                        if FS.is_file(src_file_path):
                            # Copy file using get_from and put_object_from_local_file
                            local_path = FS.get_from(src_file_path, wait_finish=True)
                            FS.put_object_from_local_file(local_path, dst_file_path)
                    
                    # Verify we have the expected files - at least the essential ones
                    text_encoder_files = FS.list_dir(text_encoder_dst)
                    has_bin_files = any(f.endswith('.bin') for f in text_encoder_files)
                    has_json_files = any(f.endswith('.json') for f in text_encoder_files)
                    
                    if has_bin_files and has_json_files:
                        solver.logger.info(f"Copied text encoder from {text_encoder_src} to {text_encoder_dst}")
                        solver.logger.info(f"Text encoder files: {text_encoder_files}")
                    else:
                        solver.logger.warning(f"Missing expected files in text encoder. Found: {text_encoder_files}")
                else:
                    solver.logger.warning(f"Text encoder path {text_encoder_src} does not exist")
            else:
                solver.logger.warning("Could not determine text encoder source path")
                
        # 4. Tokenizer - must be in tokenizer/t5-v1_1-xxl/ with spiece.model and 3 json files
        if 'tokenizer' in self.model_components and hasattr(model, 'cond_stage_model'):
            # We need to get the tokenizer path
            tokenizer_src = None
            
            # Try to get path from model
            if hasattr(model.cond_stage_model, 'tokenizer_path'):
                tokenizer_src = model.cond_stage_model.tokenizer_path
            # Check the model config
            elif hasattr(model, 'cfg') and hasattr(model.cfg, 'COND_STAGE_MODEL'):
                if 'TOKENIZER_PATH' in model.cfg.COND_STAGE_MODEL:
                    tokenizer_src = model.cfg.COND_STAGE_MODEL.TOKENIZER_PATH
            
            # Ensure it ends with t5-v1_1-xxl
            if tokenizer_src:
                if not tokenizer_src.endswith('t5-v1_1-xxl'):
                    if tokenizer_src.endswith('/'):
                        tokenizer_src = osp.join(tokenizer_src, 't5-v1_1-xxl')
                    else:
                        tokenizer_src = osp.join(tokenizer_src, 't5-v1_1-xxl')
                
                tokenizer_dst = osp.join(output_path, 'tokenizer', 't5-v1_1-xxl')
                
                # Verify source exists
                if FS.exists(tokenizer_src):
                    # Copy all files from tokenizer_src to tokenizer_dst
                    file_list = FS.list_dir(tokenizer_src)
                    for filename in file_list:
                        src_file_path = osp.join(tokenizer_src, filename)
                        dst_file_path = osp.join(tokenizer_dst, filename)
                        
                        if FS.is_file(src_file_path):
                            # Copy file using get_from and put_object_from_local_file
                            local_path = FS.get_from(src_file_path, wait_finish=True)
                            FS.put_object_from_local_file(local_path, dst_file_path)
                    
                    # Verify we have the expected files - at least the essential ones
                    tokenizer_files = FS.list_dir(tokenizer_dst)
                    has_spiece_model = any(f == 'spiece.model' for f in tokenizer_files)
                    has_json_files = sum(1 for f in tokenizer_files if f.endswith('.json'))
                    
                    if has_spiece_model and has_json_files >= 3:
                        solver.logger.info(f"Copied tokenizer from {tokenizer_src} to {tokenizer_dst}")
                        solver.logger.info(f"Tokenizer files: {tokenizer_files}")
                    else:
                        solver.logger.warning(f"Missing expected files in tokenizer. Found: {tokenizer_files}")
                else:
                    solver.logger.warning(f"Tokenizer path {tokenizer_src} does not exist")
            else:
                solver.logger.warning("Could not determine tokenizer source path")

        # Verify the model structure is as expected
        required_components = {
            'dit/ace_0.6b_512px.pth': 'DIT model file',
            'vae/vae.bin': 'VAE model file',
            'text_encoder/t5-v1_1-xxl': 'Text encoder directory',
            'tokenizer/t5-v1_1-xxl': 'Tokenizer directory'
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
        readme_path = osp.join(output_path, "README.md")
        readme_content = f"""# ACE Model - 0.6B 512px

This directory contains all components required for the ACE model:

- `dit/ace_0.6b_512px.pth`: Diffusion Transformer model
- `vae/vae.bin`: VAE model for encoding/decoding images
- `text_encoder/t5-v1_1-xxl/`: T5 text encoder (contains 5 .bin files and 2 .json files)
- `tokenizer/t5-v1_1-xxl/`: T5 tokenizer (contains spiece.model and 3 .json files)
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
        
        solver.logger.info(f"Saved all model components to {output_path}")
        return output_path

    def _push_to_huggingface(self, solver):
        """Push the model to Hugging Face Hub."""
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
            
            # Copy all files from full_output_dir to the local directory
            if FS.exists(self.full_output_dir):
                # List all components
                components = FS.list_dir(self.full_output_dir)
                solver.logger.info(f"Found {len(components)} components to upload: {components}")
                
                # Copy each component
                for component in components:
                    src_path = osp.join(self.full_output_dir, component)
                    dst_path = osp.join(local_dir, component)
                    
                    if FS.is_dir(src_path):
                        os.makedirs(dst_path, exist_ok=True)
                        # Copy directory
                        self._copy_directory_to_local(src_path, dst_path, solver)
                    else:
                        # Copy file
                        local_file = FS.get_from(src_path, wait_finish=True)
                        shutil.copy(local_file, dst_path)
            else:
                solver.logger.warning(f"Source directory {self.full_output_dir} does not exist")
                return
            
            # Upload using huggingface_hub
            api = HfApi()
            api.create_repo(
                repo_id=self.hub_model_id,
                private=self.hub_private,
                token=self.hub_token,
                exist_ok=True
            )
            
            # Upload all files in the directory
            solver.logger.info(f"Uploading {local_dir} to Hugging Face Hub as {self.hub_model_id}")
            api.upload_folder(
                folder_path=local_dir,
                repo_id=self.hub_model_id,
                token=self.hub_token
            )
            
            # Clean up
            shutil.rmtree(local_dir)
            
            solver.logger.info(f"Successfully pushed model to Hugging Face Hub: {self.hub_model_id}")
        except Exception as e:
            solver.logger.error(f"Failed to push to Hugging Face Hub: {e}")
            
    def _copy_directory_to_local(self, src, dst, solver):
        """Copy a directory from the file system to a local directory."""
        try:
            if not FS.exists(src):
                solver.logger.warning(f"Source directory {src} does not exist")
                return False
                
            # Create destination directory
            os.makedirs(dst, exist_ok=True)
            
            # List all files in source directory
            items = FS.list_dir(src)
            for item in items:
                s = osp.join(src, item)
                d = osp.join(dst, item)
                
                if FS.is_file(s):
                    # Get the file from the file system and save it locally
                    local_file = FS.get_from(s, wait_finish=True)
                    shutil.copy(local_file, d)
                elif FS.is_dir(s):
                    # Recursively copy directory
                    self._copy_directory_to_local(s, d, solver)
                    
            return True
        except Exception as e:
            solver.logger.warning(f"Error copying directory {src} to {dst}: {e}")
            return False

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            FinalModelHFHook.para_dict,
                            set_name=True)
