import os

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml

from src.utils import utils

log = utils.get_logger(__name__)


class OnCheckpointHparams(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # only do this 1 time
        if trainer.current_epoch == 0:
            file_path = f"{os.getcwd()}/hparams.yaml"
            log.info(f"Saving hparams to file_path: {file_path}")
            save_hparams_to_yaml(config_yaml=file_path, hparams=pl_module.hparams)
