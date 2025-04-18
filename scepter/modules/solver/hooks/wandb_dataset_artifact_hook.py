from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.hook import Hook
import os
import wandb
import logging
import shutil
import glob

@HOOKS.register_class()
class WandbDatasetArtifactHook(Hook):
    """
    Hook to log:
    1. All dataset CSVs (TRAIN_DATA and VAL_DATA) as wandb artifacts and files
    2. Everything in WORK_DIR to wandb
    """
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        # Defer dataset path extraction to runtime, as config object may not have TRAIN_DATA/VAL_DATA attributes at init
        self.train_csv = None
        self.val_csv = None
        self.extra_csvs = []
        self.work_dir = None
        # Set priority from config or default to 99
        self.priority = cfg.get('PRIORITY', 99)

    def before_solve(self, solver):
        # Extract dataset paths from solver's config at runtime
        cfg = solver.cfg
        
        # Store WORK_DIR for tracking
        self.work_dir = getattr(cfg, 'WORK_DIR', None)
        
        # Hardcode the CSV paths based on the specified yaml file
        # This ensures we always log these important files regardless of dataset type
        self.train_csv = "./cache/datasets/therapy_pair/images_therapist/training.csv"
        self.val_csv = "./cache/datasets/therapy_pair/images_therapist/validation.csv"
        
        # Log what we're setting
        self.logger.info(f"Setting CSV paths for logging: TRAIN={self.train_csv}, VAL={self.val_csv}")
            
        # Also check WandbValLossHook VAL_DATA for additional CSV paths
        for hook_cfg in getattr(cfg, 'TRAIN_HOOKS', []):
            if isinstance(hook_cfg, dict) and hook_cfg.get('NAME') == 'WandbValLossHook':
                if 'VAL_DATA' in hook_cfg and isinstance(hook_cfg['VAL_DATA'], dict):
                    val_csv = hook_cfg['VAL_DATA'].get('CSV_PATH')
                    if val_csv and val_csv not in [self.train_csv, self.val_csv]:
                        self.extra_csvs.append(val_csv)
                        
        # Log detailed information about what we found
        self.logger.info(f"Found CSV paths: TRAIN={self.train_csv}, VAL={self.val_csv}, EXTRA={self.extra_csvs}")
        self.logger.info(f"WORK_DIR: {self.work_dir}")
            
        # Only log on main process and if a wandb run exists
        wandb_run = wandb.run
        if wandb_run is not None and getattr(solver, 'local_rank', 0) == 0:
            # Step 1: Log CSVs as artifacts and to /table
            self._log_csvs_to_wandb(wandb_run)
            
            # Step 2: Set up WORK_DIR tracking (for when files are created later)
            if self.work_dir and os.path.exists(self.work_dir):
                # Make wandb track everything in WORK_DIR
                try:
                    wandb.save(os.path.join(self.work_dir, "*"), base_path=self.work_dir)
                    self.logger.info(f"Set up wandb to track all files in WORK_DIR: {self.work_dir}")
                except Exception as e:
                    self.logger.warning(f"Could not track WORK_DIR files with wandb.save: {e}")
                # after_solve is automatically handled by hook framework
                # Create a hook for after_solve to capture final files
                # solver.register_hook("after_solve", self.after_solve)
            
    def after_solve(self, solver):
        """Log all files in WORK_DIR after training is complete"""
        wandb_run = wandb.run
        if wandb_run is not None and getattr(solver, 'local_rank', 0) == 0:
            if self.work_dir and os.path.exists(self.work_dir):
                # Log everything in WORK_DIR as an artifact too
                artifact = wandb.Artifact('workdir_files', type='model_outputs')
                artifact.add_dir(self.work_dir)
                wandb_run.log_artifact(artifact)
                self.logger.info(f"Logged all files in WORK_DIR as an artifact: {self.work_dir}")
            
    def _log_csvs_to_wandb(self, wandb_run):
        """Log CSVs both as artifacts and as wandb Files under /table"""
        artifact = wandb.Artifact('dataset_csvs', type='dataset')
        files_added = 0
        
        # Create a "table" directory within wandb's run directory for tracking CSV files
        table_dir = os.path.join(wandb.run.dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        
        if self.train_csv and os.path.exists(self.train_csv):
            # Add to artifact
            artifact.add_file(self.train_csv)
            # Also save to wandb Files under /table
            train_csv_dest = os.path.join(table_dir, "training.csv")
            shutil.copy(self.train_csv, train_csv_dest)
            wandb.save(train_csv_dest)
            self.logger.info(f"Adding CSV to artifact and files/table: {self.train_csv}")
            files_added += 1
        elif self.train_csv:
            self.logger.warning(f"Training CSV file not found: {self.train_csv}")
            
        if self.val_csv and os.path.exists(self.val_csv):
            # Add to artifact
            artifact.add_file(self.val_csv)
            # Also save to wandb Files under /table
            val_csv_dest = os.path.join(table_dir, "validation.csv")
            shutil.copy(self.val_csv, val_csv_dest)
            wandb.save(val_csv_dest)
            self.logger.info(f"Adding CSV to artifact and files/table: {self.val_csv}")
            files_added += 1
        elif self.val_csv:
            self.logger.warning(f"Validation CSV file not found: {self.val_csv}")
            
        for i, csv_path in enumerate(self.extra_csvs):
            if os.path.exists(csv_path):
                # Add to artifact
                artifact.add_file(csv_path)
                # Also save to wandb Files under /table
                extra_csv_dest = os.path.join(table_dir, f"extra_{i}.csv")
                shutil.copy(csv_path, extra_csv_dest)
                wandb.save(extra_csv_dest)
                self.logger.info(f"Adding extra CSV to artifact and files/table: {csv_path}")
                files_added += 1
            else:
                self.logger.warning(f"Extra CSV file not found: {csv_path}")
        
        if files_added > 0:
            wandb_run.log_artifact(artifact)
            self.logger.info(f"Successfully logged {files_added} CSV files as wandb artifact and in Files/table section")
        else:
            self.logger.warning(f"No CSV files found to log as artifacts or files")
