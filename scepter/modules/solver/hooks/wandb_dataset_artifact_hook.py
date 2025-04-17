from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.hook import Hook
import os
import wandb
import logging

@HOOKS.register_class()
class WandbDatasetArtifactHook(Hook):
    """
    Hook to log all dataset CSVs (TRAIN_DATA and VAL_DATA) as wandb artifacts.
    """
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        # Defer dataset path extraction to runtime, as config object may not have TRAIN_DATA/VAL_DATA attributes at init
        self.train_csv = None
        self.val_csv = None
        self.extra_csvs = []
        # Set priority from config or default to 99
        self.priority = cfg.get('PRIORITY', 99)

    def before_solve(self, solver):
        # Extract dataset paths from solver's config at runtime
        cfg = solver.cfg
        
        # Get CSV paths with proper dictionary access to handle nested config
        if hasattr(cfg, 'TRAIN_DATA') and isinstance(cfg.TRAIN_DATA, dict):
            self.train_csv = cfg.TRAIN_DATA.get('CSV_PATH', None)
        else:
            self.logger.warning("TRAIN_DATA not found in config or not a dict")
            
        if hasattr(cfg, 'VAL_DATA') and isinstance(cfg.VAL_DATA, dict):
            self.val_csv = cfg.VAL_DATA.get('CSV_PATH', None)
        else:
            self.logger.warning("VAL_DATA not found in config or not a dict")
            
        # Also check WandbValLossHook VAL_DATA for additional CSV paths
        for hook_cfg in getattr(cfg, 'TRAIN_HOOKS', []):
            if isinstance(hook_cfg, dict) and hook_cfg.get('NAME') == 'WandbValLossHook':
                if 'VAL_DATA' in hook_cfg and isinstance(hook_cfg['VAL_DATA'], dict):
                    val_csv = hook_cfg['VAL_DATA'].get('CSV_PATH')
                    if val_csv and val_csv not in [self.train_csv, self.val_csv]:
                        self.extra_csvs.append(val_csv)
                        
        # Log detailed information about what we found
        self.logger.info(f"Found CSV paths: TRAIN={self.train_csv}, VAL={self.val_csv}, EXTRA={self.extra_csvs}")
            
        # Only log on main process and if wandb is initialized
        if hasattr(solver, 'wandb_run') and solver.wandb_run is not None and getattr(solver, 'local_rank', 0) == 0:
            artifact = wandb.Artifact('dataset_csvs', type='dataset')
            files_added = 0
            
            if self.train_csv and os.path.exists(self.train_csv):
                artifact.add_file(self.train_csv)
                self.logger.info(f"Adding CSV to artifact: {self.train_csv}")
                files_added += 1
            elif self.train_csv:
                self.logger.warning(f"Training CSV file not found: {self.train_csv}")
                
            if self.val_csv and os.path.exists(self.val_csv):
                artifact.add_file(self.val_csv)
                self.logger.info(f"Adding CSV to artifact: {self.val_csv}")
                files_added += 1
            elif self.val_csv:
                self.logger.warning(f"Validation CSV file not found: {self.val_csv}")
                
            for csv_path in self.extra_csvs:
                if os.path.exists(csv_path):
                    artifact.add_file(csv_path)
                    self.logger.info(f"Adding extra CSV to artifact: {csv_path}")
                    files_added += 1
                else:
                    self.logger.warning(f"Extra CSV file not found: {csv_path}")
            
            if files_added > 0:
                solver.wandb_run.log_artifact(artifact)
                self.logger.info(f"Successfully logged {files_added} CSV files as wandb artifact")
            else:
                self.logger.warning(f"No CSV files found to log as artifacts")
