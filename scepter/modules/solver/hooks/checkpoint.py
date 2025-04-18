# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import os.path as osp
import sys
import warnings
import time

import torch
import torch.distributed as du
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

_DEFAULT_CHECKPOINT_PRIORITY = 300


@HOOKS.register_class()
class CheckpointHook(Hook):
    """ Checkpoint resume or save hook.
    Args:
        interval (int): Save interval, by epoch.
        save_best (bool): Save the best checkpoint by a metric key, default is False.
        save_best_by (str): How to get the best the checkpoint by the metric key, default is ''.
            + means the higher the best (default).
            - means the lower the best.
            E.g. +acc@1, -err@1, acc@5(same as +acc@5)
    """
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_CHECKPOINT_PRIORITY,
            'description': 'the priority for processing!'
        },
        'INTERVAL': {
            'value': 1,
            'description': 'the interval of saving checkpoint!'
        },
        'SAVE_BEST': {
            'value': False,
            'description': 'If save the best model or not!'
        },
        'SAVE_BEST_BY': {
            'value':
            '',
            'description':
            'If save the best model, which order should be sorted, +/-!'
        },
        'DISABLE_SNAPSHOT': {
            'value': False,
            'description': 'Skip to save snapshot checkpoint.'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(CheckpointHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_CHECKPOINT_PRIORITY)
        self.interval = cfg.get('INTERVAL', 1)
        self.save_name_prefix = cfg.get('SAVE_NAME_PREFIX', 'ldm_step')
        self.save_last = cfg.get('SAVE_LAST', False)
        self.save_best = cfg.get('SAVE_BEST', False)
        self.save_best_by = cfg.get('SAVE_BEST_BY', '')
        self.push_to_hub = cfg.get('PUSH_TO_HUB', False)
        self.hub_model_id = cfg.get('HUB_MODEL_ID', None)
        self.hub_private = cfg.get('HUB_PRIVATE', False)
        self.disable_save_snapshot = cfg.get('DISABLE_SNAPSHOT', False)
        self.last_ckpt = None
        if self.save_best and not self.save_best_by:
            warnings.warn(
                "CheckpointHook: Parameter 'save_best_by' is not set, turn off save_best function."
            )
            self.save_best = False
        self.higher_the_best = True
        if self.save_best:
            if self.save_best_by.startswith('+'):
                self.save_best_by = self.save_best_by[1:]
            elif self.save_best_by.startswith('-'):
                self.save_best_by = self.save_best_by[1:]
                self.higher_the_best = False
        if self.save_best and not self.save_best_by:
            warnings.warn(
                "CheckpointHook: Parameter 'save_best_by' is not valid, turn off save_best function."
            )
            self.save_best = False
        self._last_best = None if not self.save_best else (
            sys.float_info.min if self.higher_the_best else sys.float_info.max)

    def before_solve(self, solver):
        if solver.resume_from is None:
            return
        if not FS.exists(solver.resume_from):
            solver.logger.error(f'File not exists {solver.resume_from}')
            return

        with FS.get_from(solver.resume_from, wait_finish=True) as local_file:
            solver.logger.info(f'Loading checkpoint from {solver.resume_from}')
            checkpoint = torch.load(local_file,
                                    map_location=torch.device('cpu'), weights_only=True)

        solver.load_checkpoint(checkpoint)
        if self.save_best and '_CheckpointHook_best' in checkpoint:
            self._last_best = checkpoint['_CheckpointHook_best']

    def after_iter(self, solver):
        if solver.total_iter != 0 and (
            (solver.total_iter + 1) % self.interval == 0
                or solver.total_iter == solver.max_steps - 1):
            solver.logger.info(
                f'Saving checkpoint after {solver.total_iter + 1} steps')
            save_path = osp.join(
                solver.work_dir,
                'checkpoints/{}-{}.pth'.format(self.save_name_prefix,
                                               solver.total_iter + 1))
            if not self.disable_save_snapshot:
                checkpoint = solver.save_checkpoint()
                if we.rank == 0:
                    with FS.put_to(save_path) as local_path:
                        with open(local_path, 'wb') as f:
                            torch.save(checkpoint, f)
                    
                    # Log checkpoint to wandb if available
                    try:
                        import wandb
                        if wandb.run is not None:
                            artifact_name = f"model-checkpoint-step-{solver.total_iter + 1}"
                            checkpoint_artifact = wandb.Artifact(
                                name=artifact_name,
                                type="model-checkpoint",
                                description=f"Model checkpoint at step {solver.total_iter + 1}"
                            )
                            
                            # Add metadata to the artifact
                            checkpoint_artifact.metadata = {
                                "step": solver.total_iter + 1,
                                "epoch": solver.epoch,
                                "timestamp": time.time(),
                                "is_final": solver.total_iter == solver.max_steps - 1
                            }
                            
                            # Add performance metrics if available
                            if hasattr(solver, 'iter_outputs'):
                                for key, value in solver.iter_outputs.items():
                                    if isinstance(value, (int, float)):
                                        checkpoint_artifact.metadata[f"metric_{key}"] = value
                            
                            # Add the checkpoint file to the artifact
                            checkpoint_artifact.add_file(local_path, name=osp.basename(save_path))
                            
                            # Log the artifact to wandb
                            wandb.run.log_artifact(checkpoint_artifact)
                            solver.logger.info(f"Logged checkpoint artifact to wandb: {artifact_name}")
                    except Exception as e:
                        solver.logger.warning(f"Failed to log checkpoint to wandb: {e}")
                del checkpoint

            from swift import SwiftModel
            if isinstance(solver.model, SwiftModel) or (
                    hasattr(solver.model, 'module')
                    and isinstance(solver.model.module, SwiftModel)):
                save_path = osp.join(
                    solver.work_dir,
                    'checkpoints/{}-{}'.format(self.save_name_prefix,
                                               solver.total_iter + 1))
                solver_model = solver.model.module if hasattr(solver.model, 'module') else solver.model
                if isinstance(solver_model.base_model.model, torch.distributed.fsdp.FullyShardedDataParallel):
                    full_state_dict_config = torch.distributed.fsdp.FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type(solver_model.base_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT, full_state_dict_config):
                        state_dict = solver_model.base_model.state_dict()
                    if we.rank == 0:
                        state_dict_new = {}
                        local_folder, _ = FS.map_to_local(save_path)
                        for adapter_name in solver_model.adapters.keys():
                            state_dict_adapter = solver_model.adapters[adapter_name].state_dict_callback(state_dict, adapter_name, replace_key=False)
                            state_dict_new.update(state_dict_adapter)
                        solver_model.save_pretrained(local_folder, state_dict=state_dict_new)
                        FS.put_dir_from_local_dir(local_folder, save_path)
                        
                        # Log model files to wandb if available
                        try:
                            import wandb
                            if wandb.run is not None:
                                model_artifact = wandb.Artifact(
                                    name=f"model-files-step-{solver.total_iter + 1}",
                                    type="model-files",
                                    description=f"Model files at step {solver.total_iter + 1}"
                                )
                                
                                # Add model directory to artifact
                                model_artifact.add_dir(local_folder, name="model")
                                
                                # Log the artifact to wandb
                                wandb.run.log_artifact(model_artifact)
                                solver.logger.info(f"Logged model files artifact to wandb at step {solver.total_iter + 1}")
                        except Exception as e:
                            solver.logger.warning(f"Failed to log model files to wandb: {e}")
                else:
                    if we.rank == 0:
                        local_folder, _ = FS.map_to_local(save_path)
                        solver_model.save_pretrained(local_folder)
                        FS.put_dir_from_local_dir(local_folder, save_path)
                        
                        # Log model files to wandb if available
                        try:
                            import wandb
                            if wandb.run is not None:
                                model_artifact = wandb.Artifact(
                                    name=f"model-files-step-{solver.total_iter + 1}",
                                    type="model-files",
                                    description=f"Model files at step {solver.total_iter + 1}"
                                )
                                
                                # Add model directory to artifact
                                model_artifact.add_dir(local_folder, name="model")
                                
                                # Log the artifact to wandb
                                wandb.run.log_artifact(model_artifact)
                                solver.logger.info(f"Logged model files artifact to wandb at step {solver.total_iter + 1}")
                        except Exception as e:
                            solver.logger.warning(f"Failed to log model files to wandb: {e}")
            else:
                if hasattr(solver, 'save_pretrained'):
                    save_path = osp.join(
                        solver.work_dir, 'checkpoints/{}-{}'.format(
                            self.save_name_prefix, solver.total_iter + 1))
                    local_folder, _ = FS.map_to_local(save_path)
                    FS.make_dir(local_folder)
                    ckpt, cfg = solver.save_pretrained()
                    if we.rank == 0:
                        with FS.put_to(
                                os.path.join(
                                    local_folder,
                                    'pytorch_model.bin')) as local_path:
                            with open(local_path, 'wb') as f:
                                torch.save(ckpt, f)
                        with FS.put_to(
                                os.path.join(
                                    local_folder,
                                    'configuration.json')) as local_path:
                            json.dump(cfg, open(local_path, 'w'))
                        FS.put_dir_from_local_dir(local_folder, save_path)
                    del ckpt

                    # Log model files to wandb if available
                    try:
                        import wandb
                        if wandb.run is not None:
                            model_artifact = wandb.Artifact(
                                name=f"model-files-step-{solver.total_iter + 1}",
                                type="model-files",
                                description=f"Model files at step {solver.total_iter + 1}"
                            )
                            
                            # Add model directory to artifact
                            model_artifact.add_dir(local_folder, name="model")
                            
                            # Log the artifact to wandb
                            wandb.run.log_artifact(model_artifact)
                            solver.logger.info(f"Logged model files artifact to wandb at step {solver.total_iter + 1}")
                    except Exception as e:
                        solver.logger.warning(f"Failed to log model files to wandb: {e}")

                if self.save_last and solver.total_iter == solver.max_steps - 1:
                    with FS.get_fs_client(save_path) as client:
                        last_path = osp.join(
                            solver.work_dir,
                            f'checkpoints/{self.save_name_prefix}-last')
                        client.make_link(last_path, save_path)
                self.last_ckpt = save_path

            torch.cuda.synchronize()
            if we.is_distributed:
                torch.distributed.barrier()

    def after_epoch(self, solver):
        if du.is_available() and du.is_initialized() and du.get_rank() != 0:
            return
        if (solver.epoch + 1) % self.interval == 0:
            solver.logger.info(
                f'Saving checkpoint after {solver.epoch} epochs')
            checkpoint = solver.save_checkpoint()
            if checkpoint is None or len(checkpoint) == 0:
                return
            cur_is_best = False
            if self.save_best:
                # Try to get current state from epoch_outputs["eval"]
                cur_state = None \
                    if self.save_best_by not in solver.epoch_outputs['eval'] \
                    else solver.epoch_outputs['eval'][self.save_best_by]
                # Try to get current state from agg_iter_outputs["eval"] if do_final_eval is False
                if cur_state is None:
                    cur_state = None \
                        if self.save_best_by not in solver.agg_iter_outputs['eval'] \
                        else solver.agg_iter_outputs['eval'][self.save_best_by]
                # Try to get current state from agg_iter_outputs["train"] if no evaluation
                if cur_state is None:
                    cur_state = None \
                        if self.save_best_by not in solver.agg_iter_outputs['train'] \
                        else solver.agg_iter_outputs['train'][self.save_best_by]
                if cur_state is not None:
                    if self.higher_the_best and cur_state > self._last_best:
                        self._last_best = cur_state
                        cur_is_best = True
                    elif not self.higher_the_best and cur_state < self._last_best:
                        self._last_best = cur_state
                        cur_is_best = True
                    checkpoint['_CheckpointHook_best'] = self._last_best
            # minus 1, means index
            save_path = osp.join(solver.work_dir,
                                 'epoch-{:05d}.pth'.format(solver.epoch))

            with FS.get_fs_client(save_path) as client:
                local_file = client.convert_to_local_path(save_path)
                with open(local_file, 'wb') as f:
                    torch.save(checkpoint, f)
                client.put_object_from_local_file(local_file, save_path)

                if cur_is_best:
                    best_path = osp.join(solver.work_dir, 'best.pth')
                    client.make_link(best_path, save_path)
            # save pretrain checkout
            if 'pre_state_dict' in checkpoint:
                save_path = osp.join(
                    solver.work_dir,
                    'epoch-{:05d}_pretrain.pth'.format(solver.epoch))
                with FS.get_fs_client(save_path) as client:
                    local_file = client.convert_to_local_path(save_path)
                    with open(local_file, 'wb') as f:
                        torch.save(checkpoint['pre_state_dict'], f)
                    client.put_object_from_local_file(local_file, save_path)
            del checkpoint

    def after_all_iter(self, solver):
        if we.rank == 0:
            if self.push_to_hub and self.last_ckpt:
                # Use huggingface_hub for pushing to HF
                try:
                    import sys
                    import subprocess
                    
                    # Debug information to help diagnose import issues
                    solver.logger.info(f"Python path: {sys.path}")
                    try:
                        # Try to determine where huggingface_hub is installed
                        result = subprocess.run(
                            ["python3.10", "-c", "import huggingface_hub; print(huggingface_hub.__file__)"],
                            capture_output=True, text=True, check=True
                        )
                        solver.logger.info(f"huggingface_hub located at: {result.stdout.strip()}")
                    except subprocess.CalledProcessError as e:
                        solver.logger.error(f"Could not locate huggingface_hub: {e.stderr}")
                    
                    # Try direct import
                    solver.logger.info("Attempting to import huggingface_hub...")
                    import huggingface_hub
                    from huggingface_hub import HfApi
                    
                    print("\n" + "="*80)
                    print(" ATTEMPTING TO PUSH MODEL TO HUGGING FACE HUB")
                    print("="*80 + "\n")
                    
                    with FS.get_dir_to_local_dir(self.last_ckpt) as local_dir:
                        # Get token from environment variable
                        token = os.environ.get("HUGGINGFACE_TOKEN", None)
                        if token is None:
                            print("\n" + "="*80)
                            print(" ERROR: HUGGINGFACE_TOKEN ENVIRONMENT VARIABLE NOT SET")
                            print("="*80 + "\n")
                            solver.logger.error("HUGGINGFACE_TOKEN environment variable not set. Cannot push to Hugging Face Hub.")
                            return
                        
                        print(f"\n Preparing to upload model to: {self.hub_model_id}")
                        print(f" Local directory being uploaded: {local_dir}")
                        solver.logger.info(f"Pushing model to Hugging Face Hub: {self.hub_model_id}")
                        api = HfApi(token=token)
                        api.create_repo(
                            repo_id=self.hub_model_id,
                            private=self.hub_private,
                            exist_ok=True
                        )
                        print(f" Starting upload to Hugging Face... (this may take a while)")
                        # List files being uploaded to help with debugging
                        print(f"\nUploading the following files:")
                        import os
                        for root, dirs, files in os.walk(local_dir):
                            for file in files:
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, local_dir)
                                file_size = os.path.getsize(full_path) / (1024 * 1024)  # Convert to MB
                                print(f" - {rel_path} ({file_size:.2f} MB)")
                        api.upload_folder(
                            folder_path=local_dir,
                            repo_id=self.hub_model_id,
                            repo_type="model"
                        )
                        
                        # Check what was actually uploaded by listing repo contents
                        print("\nVerifying repository contents on Hugging Face:")
                        try:
                            repo_files = api.list_repo_files(repo_id=self.hub_model_id, repo_type="model")
                            for file in repo_files:
                                print(f" - {file}")
                            print(f"\nTotal files in repository: {len(repo_files)}")
                        except Exception as e:
                            print(f"Could not list repository contents: {e}")
                        print("\n" + "="*80)
                        print(f" SUCCESS! MODEL UPLOADED TO HUGGING FACE HUB: {self.hub_model_id}")
                        print(f" View your model at: https://huggingface.co/{self.hub_model_id}")
                        print("="*80 + "\n")
                        solver.logger.info(f"Successfully pushed model to Hugging Face Hub: {self.hub_model_id}")
                except ImportError as e:
                    print("\n" + "="*80)
                    print(f" ERROR IMPORTING HUGGINGFACE_HUB: {e}")
                    print("="*80 + "\n")
                    solver.logger.error(f"huggingface_hub import error: {e}")
                    solver.logger.error("Please make sure huggingface_hub is properly installed in the current Python environment")
                except Exception as e:
                    print("\n" + "="*80)
                    print(f" ERROR PUSHING TO HUGGING FACE HUB: {e}")
                    print("="*80 + "\n")
                    solver.logger.error(f"Error pushing to Hugging Face Hub: {e}")

    @staticmethod
    def get_config_template():
        return dict_to_yaml('hook',
                            __class__.__name__,
                            CheckpointHook.para_dict,
                            set_name=True)
