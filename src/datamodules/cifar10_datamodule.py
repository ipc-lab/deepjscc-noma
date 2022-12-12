from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.datamodules.components.grouping_wrapper import GroupingWrapper


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        train_val_split=(45000, 5000),
        num_workers: int = 0,
        pin_memory: bool = True,
        anomaly_label: int = 1,
        group_size: int = 2,
        num_train_samples: int = -1,
        num_val_samples: int = -1,
        num_test_samples: int = -1,
        regroup_every_training_epoch: bool = True,
        min_snr: float = 0.0,
        max_snr: float = 20.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.counter = 1
        
    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir, train=True, transform=self.transforms)
            data_test = CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)

            self.data_train_single, data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(1),
            )
            self.num_train_samples = self.hparams.num_train_samples if self.hparams.num_train_samples != 0 else len(self.data_train_single)

            csi_train = self.generate_noise_vec(self.num_train_samples)
            csi_val = self.generate_noise_vec(len(data_val))

            self.data_train = GroupingWrapper(
                self.data_train_single,
                csi_train,
                self.hparams.group_size,
                self.hparams.num_train_samples,
                seed=1,
            )

            self.data_val = GroupingWrapper(
                data_val,
                csi_val,
                self.hparams.group_size,
                self.hparams.num_val_samples,
                seed=1,
            )

            self.data_test = [
                GroupingWrapper(
                    data_test,
                    self.generate_noise_vec(len(data_test), min_snr=snr, max_snr=snr, seed=i),
                    self.hparams.group_size,
                    self.hparams.num_test_samples,
                    seed=1,
                )
                for i, snr in enumerate(
                    torch.arange(self.hparams.min_snr, self.hparams.max_snr + 1.0, 1.0)
                )
            ]

    def train_dataloader(self):
        if self.hparams.regroup_every_training_epoch:
            csi_train = self.generate_noise_vec(self.num_train_samples, seed=self.counter)

            data_train = GroupingWrapper(
                self.data_train_single,
                csi_train,
                self.hparams.group_size,
                self.hparams.num_train_samples,
                seed=self.counter,
            )
            self.counter += 1
        else:
            data_train = self.data_train

        return DataLoader(
            dataset=data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                dataset=dt,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
            for dt in self.data_test
        ]

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def generate_noise_vec(self, num_samples, min_snr=None, max_snr=None, seed=1):

        min_snr = min_snr if min_snr is not None else self.hparams.min_snr
        max_snr = max_snr if max_snr is not None else self.hparams.max_snr

        return torch.empty(num_samples, 1).uniform_(
            min_snr,
            max_snr,
            generator=torch.Generator().manual_seed(seed),
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "cifar10.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
