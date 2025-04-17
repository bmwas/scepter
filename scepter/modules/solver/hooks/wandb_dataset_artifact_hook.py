from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.hook import Hook
import os
import wandb
import logging
import shutil

@HOOKS.register_class()
class WandbDatasetArtifactHook(Hook):
    """
    Hook to log all dataset CSVs (TRAIN_DATA and VAL_DATA) as wandb artifacts.
    Also saves the CSV files directly to wandb Files section.
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
            
            # Create a "datasets" directory within wandb's run directory for tracking files
            datasets_dir = os.path.join(wandb.run.dir, "datasets")
            os.makedirs(datasets_dir, exist_ok=True)
            
            if self.train_csv and os.path.exists(self.train_csv):
                # Add to artifact
                artifact.add_file(self.train_csv)
                # Also save to wandb Files
                train_csv_dest = os.path.join(datasets_dir, "training.csv")
                shutil.copy(self.train_csv, train_csv_dest)
                wandb.save(train_csv_dest)
                self.logger.info(f"Adding CSV to artifact and files: {self.train_csv}")
                files_added += 1
            elif self.train_csv:
                self.logger.warning(f"Training CSV file not found: {self.train_csv}")
                
            if self.val_csv and os.path.exists(self.val_csv):
                # Add to artifact
                artifact.add_file(self.val_csv)
                # Also save to wandb Files
                val_csv_dest = os.path.join(datasets_dir, "validation.csv")
                shutil.copy(self.val_csv, val_csv_dest)
                wandb.save(val_csv_dest)
                self.logger.info(f"Adding CSV to artifact and files: {self.val_csv}")
                files_added += 1
            elif self.val_csv:
                self.logger.warning(f"Validation CSV file not found: {self.val_csv}")
                
            for i, csv_path in enumerate(self.extra_csvs):
                if os.path.exists(csv_path):
                    # Add to artifact
                    artifact.add_file(csv_path)
                    # Also save to wandb Files
                    extra_csv_dest = os.path.join(datasets_dir, f"extra_{i}.csv")
                    shutil.copy(csv_path, extra_csv_dest)
                    wandb.save(extra_csv_dest)
                    self.logger.info(f"Adding extra CSV to artifact and files: {csv_path}")
                    files_added += 1
                else:
                    self.logger.warning(f"Extra CSV file not found: {csv_path}")
            
            if files_added > 0:
                solver.wandb_run.log_artifact(artifact)
                self.logger.info(f"Successfully logged {files_added} CSV files as wandb artifact and in Files section")
            else:
                self.logger.warning(f"No CSV files found to log as artifacts or files")
