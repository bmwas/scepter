import os
import shutil
from scepter.modules.solver.hooks.registry import HOOKS
from scepter.modules.solver.hooks.hook import Hook

@HOOKS.register_class()
class CopyConfigHook(Hook):
    """
    Hook to copy the training config file to a known directory (e.g., ./cache/save_data/) before training starts.
    This ensures the config is always tracked by artifact/file tracking hooks.
    """
    def before_solve(self, solver):
        src = './scepter/methods/edit/wandb_dit_ace_0.6b_512.yaml'
        dst = './cache/save_data/wandb_dit_ace_0.6b_512.yaml'
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            solver.logger.info(f"CopyConfigHook: Copied config from {src} to {dst}")
        except Exception as e:
            solver.logger.warning(f"CopyConfigHook: Failed to copy config: {e}")
