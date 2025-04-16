# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numbers
import os
import os.path as osp
import time
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import LogAgg, time_since

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:
    warnings.warn(f'Runing without tensorboard! {e}')

_DEFAULT_LOG_PRIORITY = 100


def _format_float(x):
    try:
        if abs(x) - int(abs(x)) < 0.01:
            return '{:.6f}'.format(x)
        else:
            return '{:.4f}'.format(x)
    except Exception:
        return 'NaN'


def _print_v(x):
    if isinstance(x, float):
        return _format_float(x)
    elif isinstance(x, torch.Tensor) and x.ndim == 0:
        return _print_v(x.item())
    else:
        return f'{x}'


def _print_iter_log(solver, outputs, final=False, start_time=0, mode=None):
    """Print the log info in iter."""
    if mode is None:
        mode = solver.mode

    def _print_v(v, fmt='{:.4f}'):
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, numbers.Number):
            v = fmt.format(v)
        return v

    s = []
    if outputs is None:
        print('No log is available')
        return None
    for k, v in outputs.items():
        if k == 'data_time' or k == 'time' or k == 'loss':
            continue
        if isinstance(v, list) and len(v) >= 2:
            s.append(f'{k}: {_print_v(v[0])}({_print_v(v[1])})')
        else:
            s.append(f'{k}: {_print_v(v)}')

    if 'loss' in outputs:
        v = outputs['loss']
        s.insert(0, 'loss: ' + _print_v(v[0]) + f'({_print_v(v[1])})')

    if 'time' in outputs:
        v = outputs['time']
        s.insert(0, 'time: ' + _print_v(v[0]) + f'({_print_v(v[1])})')
    if 'data_time' in outputs:
        v = outputs['data_time']
        s.insert(0, 'data_time: ' + _print_v(v[0]) + f'({_print_v(v[1])})')

    if solver.max_epochs == -1:
        assert solver.max_steps > 0
        percent = (solver.total_iter +
                   1 if not final else solver.total_iter) / solver.max_steps
        now_status = time_since(start_time, percent)
        solver.logger.info(
            f'Stage [{mode}] '
            f'iter: [{solver.total_iter + 1 if not final else solver.total_iter}/{solver.max_steps}], '
            f"{', '.join(s)}, "
            f'[{now_status}]')
    else:
        assert solver.max_epochs > 0 and solver.epoch_max_iter > 0
        percent = (solver.total_iter + 1 if not final else solver.total_iter
                   ) / (solver.epoch_max_iter * solver.max_epochs)
        now_status = time_since(start_time, percent)
        solver.logger.info(
            f'Stage [{mode}] '
            f'iter: [{solver.iter + 1 if not final else solver.iter}/{solver.epoch_max_iter}], '
            f"{', '.join(s)}, "
            f'[{now_status} {percent*100:.2f}%({time_since(start_time, 1)})]')
    
    # Also log metrics to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            # Create a dict of metrics to log to wandb
            wandb_metrics = {}
            
            # Add iteration/step info
            current_iter = solver.total_iter + (0 if final else 1)
            wandb_metrics["global_step"] = current_iter
            wandb_metrics[f"{mode}/iter_number"] = current_iter
            
            # Add detailed training progress metrics
            wandb_metrics[f"{mode}/progress_percent"] = percent * 100
            wandb_metrics[f"{mode}/epoch"] = solver.epoch
            if solver.max_epochs > 0:
                wandb_metrics[f"{mode}/epoch_progress"] = (solver.epoch + (solver.iter / solver.epoch_max_iter)) / solver.max_epochs * 100
            
            # Process all metrics from outputs with more detailed naming
            for k, v in outputs.items():
                if isinstance(v, list) and len(v) >= 2:
                    # For metrics that have current and average values (like loss, time, etc.)
                    # Log both the current value and the running average with clear naming
                    wandb_metrics[f"{mode}/{k}/current"] = _print_v(v[0], fmt='{:.6f}')
                    wandb_metrics[f"{mode}/{k}/average"] = _print_v(v[1], fmt='{:.6f}')
                    # Also log with simpler naming for backward compatibility
                    wandb_metrics[f"{mode}/{k}"] = _print_v(v[0], fmt='{:.6f}')
                    wandb_metrics[f"{mode}/{k}_avg"] = _print_v(v[1], fmt='{:.6f}')
                else:
                    # For simple metrics, just log the value
                    wandb_metrics[f"{mode}/{k}"] = _print_v(v, fmt='{:.6f}')
            
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
                        wandb_metrics[f"system/gpu{i}/memory_allocated_mb"] = mem_allocated
                        wandb_metrics[f"system/gpu{i}/memory_reserved_mb"] = mem_reserved
                except:
                    pass
            
            # Add throughput metrics if available
            if 'throughput' in outputs:
                # Check if we have the numeric value stored separately
                if 'throughput_numeric' in outputs:
                    wandb_metrics[f"{mode}/throughput"] = outputs['throughput_numeric']
                    wandb_metrics[f"{mode}/throughput/samples_per_day"] = outputs['throughput_numeric']
                # Convert string format "X/day" to numeric value
                elif isinstance(outputs['throughput'], str) and '/day' in outputs['throughput']:
                    try:
                        throughput_value = int(outputs['throughput'].replace('/day', '').strip())
                        wandb_metrics[f"{mode}/throughput"] = throughput_value
                        wandb_metrics[f"{mode}/throughput/samples_per_day"] = throughput_value
                    except ValueError:
                        # If conversion fails, log the original string
                        wandb_metrics[f"{mode}/throughput_str"] = outputs['throughput']
                else:
                    # If it's already a numeric value
                    wandb_metrics[f"{mode}/throughput"] = outputs['throughput']
                    wandb_metrics[f"{mode}/throughput/samples_per_day"] = outputs['throughput']
                
                # Handle all_throughput
                if 'all_throughput' in outputs:
                    # Ensure it's a numeric value
                    if isinstance(outputs['all_throughput'], (int, float, np.number)):
                        wandb_metrics[f"{mode}/all_throughput"] = float(outputs['all_throughput'])
                    else:
                        try:
                            wandb_metrics[f"{mode}/all_throughput"] = float(outputs['all_throughput'])
                        except (ValueError, TypeError):
                            # If conversion fails, log it to a different key
                            wandb_metrics[f"{mode}/all_throughput_str"] = str(outputs['all_throughput'])
            
            # Handle data_time specifically - ensure it's logged as a scalar
            if 'data_time' in outputs:
                if isinstance(outputs['data_time'], list) and len(outputs['data_time']) > 0:
                    # Log the current value (first element)
                    current_data_time = outputs['data_time'][0]
                    if isinstance(current_data_time, (str, int, float, np.number)):
                        try:
                            wandb_metrics[f"{mode}/data_time"] = float(current_data_time)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(current_data_time, torch.Tensor):
                        wandb_metrics[f"{mode}/data_time"] = current_data_time.item()
            
            # Log estimated time remaining
            wandb_metrics[f"{mode}/time_remaining_seconds"] = (1 - percent) * (time.time() - start_time) / max(percent, 1e-8)
            
            # Log to wandb
            wandb.log(wandb_metrics, step=current_iter)
    except Exception as e:
        # Log error but continue execution
        solver.logger.warning(f"Error logging to wandb in _print_iter_log: {e}")


def print_memory_status():
    if torch.cuda.is_available():
        nvi_info = os.popen('nvidia-smi').read()
        gpu_mem = nvi_info.split('\n')[9].split('|')[2].split('/')[0].strip()
        gpu_mem = int(gpu_mem.replace('MiB', ''))
    else:
        gpu_mem = 0
    return gpu_mem


@HOOKS.register_class()
class LogHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LOG_PRIORITY,
            'description': 'the priority for processing!'
        },
        'LOG_INTERVAL': {
            'value': 10,
            'description': 'the interval for log print!'
        },
        'SHOW_GPU_MEM': {
            'value': False,
            'description': 'to show the gpu memory'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(LogHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_LOG_PRIORITY)
        self.log_interval = cfg.get('LOG_INTERVAL', 10)
        self.interval = cfg.get('INTERVAL', self.log_interval)
        self.show_gpu_mem = cfg.get('SHOW_GPU_MEM', False)
        self.log_agg_dict = defaultdict(LogAgg)

        self.last_log_step = ('train', 0)

        self.time = time.time()
        self.start_time = time.time()
        self.all_throughput = 0
        self.data_time = 0
        self.batch_size = defaultdict(dict)

    def before_all_iter(self, solver):
        self.time = time.time()
        self.last_log_step = (solver.mode, 0)
        if hasattr(solver, "datas"):
            for k, v in solver.datas.items():
                if hasattr(v, 'batch_size'):
                    self.batch_size[k] = v.batch_size
    def before_iter(self, solver):
        data_time = time.time() - self.time
        self.data_time = data_time

    def after_iter(self, solver):
        """Log metrics after each iteration."""
        if we.rank != 0:
            return

        # Log at every iteration
        log_agg = self.log_agg_dict[solver.mode]
        iter_time = time.time() - self.time
        self.time = time.time()
        outputs = solver.iter_outputs.copy()
        outputs['time'] = iter_time
        outputs['data_time'] = self.data_time
        if solver.mode in self.batch_size:
            # Calculate throughput as numeric value
            throughput_value = int(self.batch_size[solver.mode] * we.data_group_world_size / iter_time * 86400)
            # Store raw numeric value for wandb logging
            outputs['throughput'] = throughput_value
            # Store formatted string version for display in logs
            outputs['throughput_display'] = f"{throughput_value}/day"
        log_agg.update(outputs, 1)
        log_agg_result = log_agg.aggregate(1)  # Aggregate with interval of 1 to log at every iteration
        
        # Format throughput for display in logs, but keep the numeric value for wandb
        if 'throughput' in log_agg_result:
            # Store the numeric value for wandb in a separate key
            log_agg_result['throughput_numeric'] = int(log_agg_result['throughput'][-1])
            # Format the display version
            log_agg_result['throughput'] = f"{int(log_agg_result['throughput'][-1])}/day"
        
        if solver.mode in self.batch_size:
            # Store as numeric value for wandb
            log_agg_result['all_throughput'] = (solver.iter + 1) * we.data_group_world_size * self.batch_size[solver.mode]
        
        if self.show_gpu_mem:
            log_agg_result['nvidia-smi'] = str(print_memory_status()) +"MiB"

        if solver.total_iter == 0 or \
            self.last_log_step[0] != solver.mode or \
                (solver.total_iter - self.last_log_step[1]
                 ) % self.log_interval == 0:
            self.last_log_step = (solver.mode, solver.total_iter)
            outputs = {}
            outputs.update(solver.iter_outputs)
            outputs.update(solver.collect_log_vars())
            
            # Try direct wandb logging without disrupting normal flow
            if solver.total_iter % 10 == 0:  # Only log every 10 iterations
                try:
                    import wandb
                    if wandb.run is not None and 'loss' in outputs:
                        loss_value = outputs['loss']
                        # Convert to scalar if it's a list or tensor
                        if isinstance(loss_value, torch.Tensor):
                            if loss_value.numel() == 1:
                                loss_value = loss_value.item()
                            else:
                                loss_value = loss_value.mean().item()
                        elif isinstance(loss_value, list) and len(loss_value) > 0:
                            loss_value = sum(float(v) for v in loss_value) / len(loss_value)
                            
                        # Log the scalar value
                        wandb.log({
                            "direct/loss": loss_value,
                            "direct/step": solver.total_iter
                        }, step=solver.total_iter)
                        print(f"Direct wandb log: loss={loss_value}")
                except Exception as e:
                    print(f"Wandb direct logging failed: {e}")
            
            # Original logging functionality
            _print_iter_log(
                solver, outputs, final=False, start_time=self.start_time)
        self.last_log_step = (solver.mode, solver.total_iter)

    def after_all_iter(self, solver):
        outputs = self.log_agg_dict[solver.mode].aggregate(
            solver.iter - self.last_log_step[1])
        solver.agg_iter_outputs = {
            key: value[1]
            for key, value in outputs.items()
        }
        current_log_step = (solver.mode, solver.iter)
        if current_log_step != self.last_log_step:
            _print_iter_log(solver,
                            outputs,
                            final=True,
                            start_time=self.start_time,
                            mode=solver.mode)
            self.last_log_step = current_log_step

        for _, value in self.log_agg_dict.items():
            value.reset()

    def after_epoch(self, solver):
        outputs = solver.epoch_outputs
        mode_s = []
        for mode_name, kvs in outputs.items():
            if len(kvs) == 0:
                return
            s = [f'{k}: ' + _print_v(v) for k, v in kvs.items()]
            mode_s.append(f"{mode_name} -> {', '.join(s)}")
        if len(mode_s) > 1:
            states = '\n\t'.join(mode_s)
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], \n\t'
                f'{states}')
        elif len(mode_s) == 1:
            solver.logger.info(
                f'Epoch [{solver.epoch}/{solver.max_epochs}], {mode_s[0]}')
        # summary

        for mode in self.log_agg_dict:
            solver.logger.info(f'Current Epoch {mode} Summary:')
            log_agg = self.log_agg_dict[mode]
            _print_iter_log(solver,
                            log_agg.aggregate(self.interval),
                            start_time=self.start_time,
                            mode=mode)
            if not mode == 'train':
                self.log_agg_dict[mode].reset()

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            LogHook.para_dict,
                            set_name=True)


@HOOKS.register_class()
class TensorboardLogHook(Hook):
    para_dict = [{
        'PRIORITY': {
            'value': _DEFAULT_LOG_PRIORITY,
            'description': 'the priority for processing!'
        },
        'LOG_DIR': {
            'value': None,
            'description': 'the dir for tensorboard log!'
        },
        'LOG_INTERVAL': {
            'value': 10000,
            'description': 'the interval for log upload!'
        }
    }]

    def __init__(self, cfg, logger=None):
        super(TensorboardLogHook, self).__init__(cfg, logger=logger)
        self.priority = cfg.get('PRIORITY', _DEFAULT_LOG_PRIORITY)
        self.log_dir = cfg.get('LOG_DIR', None)
        self.log_interval = cfg.get('LOG_INTERVAL', 1000)
        self.interval = cfg.get('INTERVAL', self.log_interval)
        self._local_log_dir = None
        self.writer: Optional[SummaryWriter] = None

    def before_solve(self, solver):
        if we.rank != 0:
            return

        if self.log_dir is None:
            self.log_dir = osp.join(solver.work_dir, 'tensorboard')

        self._local_log_dir, _ = FS.map_to_local(self.log_dir)
        os.makedirs(self._local_log_dir, exist_ok=True)
        self.writer = SummaryWriter(self._local_log_dir)
        solver.logger.info(f'Tensorboard: save to {self.log_dir}')

    def after_iter(self, solver):
        if self.writer is None:
            return
        outputs = solver.iter_outputs.copy()
        extra_vars = solver.collect_log_vars()
        outputs.update(extra_vars)
        mode = solver.mode
        for key, value in outputs.items():
            if key == 'batch_size':
                continue
            if isinstance(value, torch.Tensor):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = value.item()
            elif isinstance(value, np.ndarray):
                # Must be scalar
                if not value.ndim == 0:
                    continue
                value = float(value)
            elif isinstance(value, numbers.Number):
                # Must be number
                pass
            else:
                continue

            self.writer.add_scalar(f'{mode}/iter/{key}',
                                   value,
                                   global_step=solver.total_iter)
        if solver.total_iter % self.interval:
            self.writer.flush()
            # Put to remote file systems every epoch
            FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_epoch(self, solver):
        if self.writer is None:
            return
        outputs = solver.epoch_outputs.copy()
        for mode, kvs in outputs.items():
            for key, value in kvs.items():
                self.writer.add_scalar(f'{mode}/epoch/{key}',
                                       value,
                                       global_step=solver.epoch)

        self.writer.flush()
        # Put to remote file systems every epoch
        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    def after_solve(self, solver):
        if self.writer is None:
            return
        if self.writer:
            self.writer.close()

        FS.put_dir_from_local_dir(self._local_log_dir, self.log_dir)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('HOOK',
                            __class__.__name__,
                            TensorboardLogHook.para_dict,
                            set_name=True)
