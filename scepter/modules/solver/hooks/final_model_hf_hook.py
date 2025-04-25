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
            # Only push to Hugging Face Hub if configured, and do NOT log to wandb
            if self.push_to_hub and self.hub_model_id:
                self._push_to_huggingface(solver)
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
        
        # Use the new robust copy/update method
        self._copy_original_and_update(solver, output_path)
        # Do NOT log to wandb here; only export for HF
        solver.logger.info(f"Model export complete at {output_path}")
        
    def _copy_original_and_update(self, solver, output_path):
        """
        Copy the original Hugging Face model directory structure to output_path/models, then overwrite only the updated weights.
        """
        import shutil
        import os
        import os.path as osp
        import torch
        
        # Always use /models as the root for all components
        models_dir = osp.join(output_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # 1. Parse original model paths from config
        dit_src = solver.cfg.MODEL.DIFFUSION_MODEL.PRETRAINED_MODEL
        vae_src = solver.cfg.MODEL.FIRST_STAGE_MODEL.PRETRAINED_MODEL
        text_encoder_src = solver.cfg.MODEL.COND_STAGE_MODEL.PRETRAINED_MODEL
        tokenizer_src = solver.cfg.MODEL.COND_STAGE_MODEL.TOKENIZER_PATH
        
        # Helper to copy directories or files
        def copy_dir_or_file(src, dst):
            if src.endswith(('.pth', '.bin')):
                with FS.get_to_local_file(src) as local_file:
                    os.makedirs(osp.dirname(dst), exist_ok=True)
                    shutil.copy2(local_file, dst)
            else:
                with FS.get_dir_to_local_dir(src, wait_finish=True) as local_dir:
                    if not osp.exists(dst):
                        shutil.copytree(local_dir, dst)
                    else:
                        for item in os.listdir(local_dir):
                            s = osp.join(local_dir, item)
                            d = osp.join(dst, item)
                            if osp.isdir(s):
                                if not osp.exists(d):
                                    shutil.copytree(s, d)
                            else:
                                shutil.copy2(s, d)
        # Copy DIT
        dit_dst_dir = osp.join(models_dir, "dit")
        os.makedirs(dit_dst_dir, exist_ok=True)
        copy_dir_or_file(osp.dirname(dit_src), dit_dst_dir)
        # Copy VAE
        vae_dst_dir = osp.join(models_dir, "vae")
        os.makedirs(vae_dst_dir, exist_ok=True)
        copy_dir_or_file(osp.dirname(vae_src), vae_dst_dir)
        # Copy text_encoder
        text_encoder_dst_dir = osp.join(models_dir, "text_encoder", "t5-v1_1-xxl")
        os.makedirs(text_encoder_dst_dir, exist_ok=True)
        copy_dir_or_file(text_encoder_src, text_encoder_dst_dir)
        # Copy tokenizer
        tokenizer_dst_dir = osp.join(models_dir, "tokenizer", "t5-v1_1-xxl")
        os.makedirs(tokenizer_dst_dir, exist_ok=True)
        copy_dir_or_file(tokenizer_src, tokenizer_dst_dir)
        
        # 2. Overwrite only updated weights using correct model attributes
        model = solver.model.module if hasattr(solver.model, 'module') else solver.model
        # DIT (diffusion transformer)
        dit_weights_path = osp.join(dit_dst_dir, "ace_0.6b_512px.pth")
        torch.save(model.model.state_dict(), dit_weights_path)
        # VAE
        vae_weights_path = osp.join(vae_dst_dir, "vae.bin")
        torch.save(model.first_stage_model.state_dict(), vae_weights_path)
        # Text Encoder (if use_grad)
        use_grad = getattr(model.cond_stage_model, 'use_grad', False)
        if use_grad:
            text_encoder_weights_path = osp.join(text_encoder_dst_dir, "pytorch_model.bin")
            torch.save(model.cond_stage_model.state_dict(), text_encoder_weights_path)
        # Tokenizer is never updated
        solver.logger.info(f"Original model copied and updated weights saved to {models_dir}")

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
            models_dir = osp.join(dst_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            os.makedirs(osp.join(models_dir, "dit"), exist_ok=True)
            os.makedirs(osp.join(models_dir, "vae"), exist_ok=True)
            os.makedirs(osp.join(models_dir, "text_encoder", "t5-v1_1-xxl"), exist_ok=True)
            os.makedirs(osp.join(models_dir, "tokenizer", "t5-v1_1-xxl"), exist_ok=True)
            
            # Copy main files (README.md, config.yaml)
            readme_src = osp.join(src_dir, "README.md")
            config_src = osp.join(src_dir, "config.yaml")
            
            # Copy README if it exists
            if FS.exists(readme_src):
                readme_dst = osp.join(models_dir, "README.md")
                self._copy_file_for_upload(readme_src, readme_dst, solver)
            
            # Copy config if it exists
            if FS.exists(config_src):
                config_dst = osp.join(models_dir, "config.yaml")
                self._copy_file_for_upload(config_src, config_dst, solver)
            
            # Copy DIT model
            dit_src = osp.join(src_dir, "dit", "ace_0.6b_512px.pth")
            if FS.exists(dit_src):
                dit_dst = osp.join(models_dir, "dit", "ace_0.6b_512px.pth")
                self._copy_file_for_upload(dit_src, dit_dst, solver)
            else:
                solver.logger.warning(f"DIT model file not found: {dit_src}")
            
            # Copy VAE model
            vae_src = osp.join(src_dir, "vae", "vae.bin")
            if FS.exists(vae_src):
                vae_dst = osp.join(models_dir, "vae", "vae.bin") 
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
                    dst_file = osp.join(models_dir, "text_encoder", "t5-v1_1-xxl", filename)
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
                    dst_file = osp.join(models_dir, "tokenizer", "t5-v1_1-xxl", filename)
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
