from typing import Optional, Tuple

from numpy.core.numeric import full
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split

from src.datamodules.datasets import create_dataloader, create_dataset
from src.utils import utils

log = utils.get_logger(__name__)


class DefaultDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        self.data_train = create_dataset(self.opt.train)
        log.info(
            f"Train dataset [{type(self.data_train).__name__}] of size {len(self.data_train)} created."
        )
        self.data_val = create_dataset(self.opt.val)
        log.info(
            f"Val dataset [{type(self.data_val).__name__}] of size {len(self.data_val)} created."
        )
        if self.opt.get("test", None):
            self.data_test = create_dataset(self.opt.test)
            log.info(
                f"Test dataset [{type(self.data_test).__name__}] of size {len(self.data_test)} created."
            )

    def train_dataloader(self):
        dl_train = create_dataloader(self.data_train, self.opt.train, "train")
        log.info(f"Train dataloader of size {len(dl_train)} created.")
        return dl_train

    def val_dataloader(self):
        dl_val = create_dataloader(self.data_val, self.opt.val, "val")
        log.info(f"Val dataloader of size {len(dl_val)} created.")
        return dl_val

    def test_dataloader(self):
        dl_test = create_dataloader(self.data_test, self.opt.test, "test")
        log.info(f"Test dataloader of size {len(dl_test)} created.")
        return dl_test
