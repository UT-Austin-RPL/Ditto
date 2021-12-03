from torch.utils.data import DataLoader

from src.datamodules.datasets.geo_art_dataset_v0 import GeoArtDatasetV0
from src.datamodules.datasets.geo_art_dataset_v1 import GeoArtDatasetV1


def create_dataset(dataset_opt):
    ds = eval(dataset_opt["dataset_type"])(dataset_opt)
    return ds


def create_dataloader(dataset, dataset_opt, phase):
    if phase == "train":
        return DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=True,
            num_workers=dataset_opt["num_workers"],
            sampler=None,
            drop_last=True,
            pin_memory=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            sampler=None,
            pin_memory=True,
        )
