import wandb
import os
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.base import Hook

@HOOKS.register_class()
class WandbConfigLogHook(Hook):
    """
    Logs the config file to wandb as soon as training begins.
    """
    def before_solve(self, solver):
        config_path = './scepter/methods/edit/wandb_dit_ace_0.6b_512.yaml'
        if os.path.exists(config_path):
            artifact = wandb.Artifact("run_config", type="config")
            artifact.add_file(config_path)
            wandb.run.log_artifact(artifact)
            solver.logger.info(f"WandbConfigLogHook: Logged config file {config_path} to wandb as artifact.")
        else:
            solver.logger.warning(f"WandbConfigLogHook: Config file {config_path} not found!")
