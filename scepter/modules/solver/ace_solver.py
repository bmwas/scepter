# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from tqdm import tqdm
import time
import math
import traceback

from scepter.modules.utils.data import transfer_data_to_cuda
from scepter.modules.utils.distribute import we
from scepter.modules.utils.probe import ProbeData

from .diffusion_solver import LatentDiffusionSolver
from .registry import SOLVERS


@SOLVERS.register_class()
class ACESolver(LatentDiffusionSolver):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.log_train_num = cfg.get('LOG_TRAIN_NUM', -1)
        # Configuration validation
        try:
            if hasattr(cfg.MODEL, 'TUNER') and cfg.MODEL.TUNER:
                tuner_name = cfg.MODEL.TUNER.get('NAME', None)
                self.logger.info("📋 LoRA config detected: %s", tuner_name)
                if tuner_name == 'SwiftLoRA':
                    self.logger.info("🔧 LoRA parameters: r=%s, alpha=%s, dropout=%s", 
                                  cfg.MODEL.TUNER.get('R', 'unknown'),
                                  cfg.MODEL.TUNER.get('LORA_ALPHA', 'unknown'),
                                  cfg.MODEL.TUNER.get('LORA_DROPOUT', 'unknown'))
                    target_modules = cfg.MODEL.TUNER.get('TARGET_MODULES', [])
                    self.logger.info("🎯 LoRA target modules: %s", target_modules)
                    if not target_modules:
                        self.logger.warning("⚠️ No TARGET_MODULES specified for LoRA - check your config!")
                else:
                    self.logger.info("ℹ️ Non-SwiftLoRA tuner detected: %s", tuner_name)
            else:
                self.logger.warning("⚠️ No TUNER configuration found - LoRA will NOT be applied!")
        except Exception as e:
            self.logger.error("❌ Error checking LoRA configuration: %s", str(e))
            self.logger.error("💥 LoRA initialization error details: %s", traceback.format_exc())

    def save_results(self, results):
        log_data, log_label = [], []
        for result in results:
            ret_images, ret_labels = [], []
            edit_image = result.get('edit_image', None)
            edit_mask = result.get('edit_mask', None)
            if edit_image is not None:
                for i, edit_img in enumerate(result['edit_image']):
                    if edit_img is None:
                        continue
                    ret_images.append(
                        (edit_img.permute(1, 2, 0).cpu().numpy() * 255).astype(
                            np.uint8))
                    ret_labels.append(f'edit_image{i}; ')
                    if edit_mask is not None:
                        ret_images.append(
                            (edit_mask[i].permute(1, 2, 0).cpu().numpy() *
                             255).astype(np.uint8))
                        ret_labels.append(f'edit_mask{i}; ')

            target_image = result.get('target_image', None)
            target_mask = result.get('target_mask', None)
            if target_image is not None:
                ret_images.append(
                    (target_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8))
                ret_labels.append('target_image; ')
                if target_mask is not None:
                    ret_images.append(
                        (target_mask.permute(1, 2, 0).cpu().numpy() *
                         255).astype(np.uint8))
                    ret_labels.append('target_mask; ')

            reconstruct_image = result.get('reconstruct_image', None)
            if reconstruct_image is not None:
                ret_images.append(
                    (reconstruct_image.permute(1, 2, 0).cpu().numpy() *
                     255).astype(np.uint8))
                ret_labels.append(f"{result['instruction']}")
            log_data.append(ret_images)
            log_label.append(ret_labels)
        return log_data, log_label

    @torch.no_grad()
    def run_eval(self):
        self.logger.info("=" * 80)
        self.logger.info("🔄 Starting validation process at step %d", self.total_iter)
        self.logger.info("=" * 80)
        self.eval_mode()
        
        # Adapter activation with comprehensive error handling
        try:
            if hasattr(self.model, 'set_adapter'):
                current_adapters = getattr(self.model, 'active_adapters', None)
                self.logger.info("🔍 Current LoRA adapter status: %s", 
                               "ACTIVE" if current_adapters else "INACTIVE")
                
                # Only set adapter if not already active
                if current_adapters != 'default':
                    try:
                        self.logger.info("⚙️ Attempting to set 'default' LoRA adapter...")
                        self.model.set_adapter('default')
                        self.logger.info("✅ SUCCESS: 'default' LoRA adapter activated for validation!")
                        
                        # Add details about adapter weights if possible
                        if hasattr(self.model, 'get_adapter_state_dict'):
                            try:
                                adapter_dict = self.model.get_adapter_state_dict()
                                num_params = sum(p.numel() for p in adapter_dict.values())
                                self.logger.info("📊 LoRA adapter has %d parameters across %d layers", 
                                               num_params, len(adapter_dict))
                            except Exception as pex:
                                self.logger.debug("📊 Could not count LoRA parameters: %s", str(pex))
                        
                    except Exception as e:
                        self.logger.error("❌ FAILED to set LoRA adapter: %s", str(e))
                        self.logger.error("💡 TIP: Ensure the model was correctly initialized with LoRA config")
                        # Continue with validation despite adapter error
                else:
                    self.logger.info("✅ LoRA 'default' adapter already active - validation will use LoRA weights")
            else:
                self.logger.warning("⚠️ Model lacks LoRA adapter interface - VALIDATION WILL NOT USE LORA WEIGHTS!")
                self.logger.warning("💡 TIP: Ensure Swift LoRA was properly initialized in model construction")
        except Exception as e:
            self.logger.error("❌ Unexpected error during adapter setup: %s", str(e))
            self.logger.error("💥 Adapter error details: %s", traceback.format_exc())
            # Continue with validation despite error
        
        self.before_all_iter(self.hooks_dict[self._mode])
        all_results = []
        for batch_idx, batch_data in tqdm(
                enumerate(self.datas[self._mode].dataloader)):
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                         batch_idx,
                                         step=self.total_iter,
                                         rank=we.rank)
                all_results.extend(results)
            self.after_iter(self.hooks_dict[self._mode])
        log_data, log_label = self.save_results(all_results)
        self.register_probe({'eval_label': log_label})
        self.register_probe({
            'eval_image':
            ProbeData(log_data,
                      is_image=True,
                      build_html=True,
                      build_label=log_label)
        })
        self.after_all_iter(self.hooks_dict[self._mode])

    @torch.no_grad()
    def run_test(self):
        self.test_mode()
        self.before_all_iter(self.hooks_dict[self._mode])
        all_results = []
        for batch_idx, batch_data in tqdm(
                enumerate(self.datas[self._mode].dataloader)):
            self.before_iter(self.hooks_dict[self._mode])
            if self.sample_args:
                batch_data.update(self.sample_args.get_lowercase_dict())
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                results = self.run_step_eval(transfer_data_to_cuda(batch_data),
                                         batch_idx,
                                         step=self.total_iter,
                                         rank=we.rank)
                all_results.extend(results)
            self.after_iter(self.hooks_dict[self._mode])
        log_data, log_label = self.save_results(all_results)
        self.register_probe({'test_label': log_label})
        self.register_probe({
            'test_image':
            ProbeData(log_data,
                      is_image=True,
                      build_html=True,
                      build_label=log_label)
        })

        self.after_all_iter(self.hooks_dict[self._mode])

    def run_step_val(self, batch_data, noise_generator=None):
        # Adapter verification
        try:
            current_adapters = getattr(self.model, 'active_adapters', None)
            if current_adapters != 'default':
                self.logger.warning("⚠️ LoRA ADAPTER CHECK FAILED: Expected 'default', got '%s'", current_adapters)
                self.logger.warning("💥 Validation is likely NOT USING LORA WEIGHTS!")
            else:
                self.logger.debug("✅ LoRA adapter verification: 'default' active and ready")
        except Exception as e:
            self.logger.error("❌ Error checking LoRA adapter status: %s", str(e))
        
        # Time the validation step
        start_time = time.time()
        
        # Original implementation
        loss_dict = {}
        batch_data = transfer_data_to_cuda(batch_data)
        # Remove all meta fields and fields that may be passed explicitly to diffusion.loss
        exclude_fields = [
            'sample_id', 'edit_type', 'data_type', 't', 'x_0', 'model', 'model_kwargs', 'reduction', 'noise'
        ]
        batch_data_for_model = {k: v for k, v in batch_data.items() if k not in exclude_fields}
        
        # Check for missing adapter-related inputs
        if 'prompt' in batch_data_for_model:
            prompt_format = batch_data_for_model['prompt']
            if isinstance(prompt_format, list) and len(prompt_format) > 0:
                self.logger.debug("✅ Prompt format check passed: %s", type(prompt_format[0]))
            else:
                self.logger.warning("⚠️ Unusual prompt format detected: %s", type(prompt_format))
        
        with torch.autocast(device_type='cuda', enabled=self.use_amp, dtype=self.dtype):
            try:
                if hasattr(self.model, 'module'):
                    results = self.model.module.forward_train(**batch_data_for_model)
                else:
                    results = self.model.forward_train(**batch_data_for_model)
                
                loss = results['loss']
                
                # Check if loss is suspiciously high or NaN, often indicating adapter issues
                if isinstance(loss, torch.Tensor):
                    loss_value = loss.item()
                    if math.isnan(loss_value):
                        self.logger.error("❌ VALIDATION ERROR: Loss is NaN - check LoRA adapter!")
                    elif loss_value > 1000:
                        self.logger.warning("⚠️ Unusually high loss (%f) - potential LoRA adapter issue", loss_value)
                    elif loss_value < 0.001:
                        self.logger.warning("⚠️ Unusually low loss (%f) - check validation data", loss_value)
                
                for sample_id in batch_data['sample_id']:
                    loss_dict[sample_id] = loss.detach().cpu().numpy()
            except Exception as e:
                self.logger.error("❌ VALIDATION FORWARD PASS FAILED: %s", str(e))
                self.logger.error("💥 Error during validation: %s", traceback.format_exc())
                # Return empty dict to allow training to continue
                for sample_id in batch_data['sample_id']:
                    loss_dict[sample_id] = np.array([float('nan')])
        
        # Log timing information
        latency = time.time() - start_time
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss) if isinstance(loss, (int, float)) else float('nan')
        
        if math.isnan(loss_value):
            self.logger.error("❌ Validation step failed with NaN loss in %.2fs", latency)
        else:
            self.logger.debug("⏱️ Validation step completed in %.2fs with loss=%.4f", latency, loss_value)
        
        return loss_dict

    @property
    def probe_data(self):
        if not we.debug and self.mode == 'train':
            batch_data = transfer_data_to_cuda(
                self.current_batch_data[self.mode])
            self.eval_mode()
            with torch.autocast(device_type='cuda',
                                enabled=self.use_amp,
                                dtype=self.dtype):
                batch_data['log_num'] = self.log_train_num
                results = self.run_step_eval(batch_data)
            self.train_mode()
            log_data, log_label = self.save_results(results)
            self.register_probe({
                'train_image':
                ProbeData(log_data,
                          is_image=True,
                          build_html=True,
                          build_label=log_label)
            })
            self.register_probe({'train_label': log_label})
        return super(LatentDiffusionSolver, self).probe_data
