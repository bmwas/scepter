import os
import wandb
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.hook import Hook

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

    def before_solve(self, solver):
        # Extract dataset paths from solver's config at runtime
        cfg = solver.cfg
        self.train_csv = getattr(cfg, 'TRAIN_DATA', {}).get('CSV_PATH', None)
        self.val_csv = getattr(cfg, 'VAL_DATA', {}).get('CSV_PATH', None)
        # Only log on main process and if wandb is initialized
        if hasattr(solver, 'wandb_run') and solver.wandb_run is not None and getattr(solver, 'local_rank', 0) == 0:
            artifact = wandb.Artifact('dataset_csvs', type='dataset')
            if self.train_csv and os.path.exists(self.train_csv):
                artifact.add_file(self.train_csv)
            if self.val_csv and os.path.exists(self.val_csv):
                artifact.add_file(self.val_csv)
            for csv_path in self.extra_csvs:
                if os.path.exists(csv_path):
                    artifact.add_file(csv_path)
            solver.wandb_run.log_artifact(artifact)
            if self.logger:
                self.logger.info(f"Logged dataset CSVs as wandb artifact: {[self.train_csv, self.val_csv] + self.extra_csvs}")
