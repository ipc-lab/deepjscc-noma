from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn

from src.utils.psnr import PSNR


class DeepJSCCModule(LightningModule):
    def __init__(
        self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, num_devices: int, N, M, loss
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.train_psnr = PSNR()
        self.val_psnr = PSNR()
        self.test_psnr = PSNR()
        
        self.device_psnr = nn.ModuleDict({f"psnr_dev{i}": PSNR() for i in range(self.hparams.num_devices)})

        self.mse_loss = nn.MSELoss()

    def loss(self, y_hat, y):

        if self.hparams.loss == "mean_mse":
            return self.mse_loss(y_hat.flatten(0, 1), y.flatten(0, 1))

        elif self.hparams.loss == "sum_mse":
            return torch.stack(
                [
                    self.mse_loss(y_hat[:, i, ...], y[:, i, ...])
                    for i in range(self.hparams.num_devices)
                ]
            ).sum()
        elif self.hparams.loss == "geometric_mse":
            return torch.sqrt(
                torch.prod(
                    torch.stack(
                        [
                            self.mse_loss(y_hat[:, i, ...], y[:, i, ...])
                            for i in range(self.hparams.num_devices)
                        ]
                    )
                )
            )

    def forward(self, x: torch.Tensor):

        return self.net(x)

    def step(self, batch: Any):
        x, csi = batch

        x_hat = self.forward((x, csi))

        additional_loss = 0.0

        if isinstance(x_hat, tuple):
            x_hat, additional_loss = x_hat

        loss = self.loss(x_hat, x) + additional_loss

        return loss, x_hat.flatten(0, 1), x.flatten(0, 1)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        psnr = self.train_psnr(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        psnr = self.val_psnr(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        psnr = self.test_psnr(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/psnr", psnr, on_step=False, on_epoch=True)
        
        # log device metrics
        num_items = preds.size(0) // self.hparams.num_devices
        for i in range(self.hparams.num_devices):
            dev_psnr = self.device_psnr[f"psnr_dev{i}"](preds[i*num_items:(i+1)*num_items, ...], targets[i*num_items:(i+1)*num_items, ...])
            self.log(f"test/psnr_dev{i}", dev_psnr, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):

        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "deepjscc.yaml")
    _ = hydra.utils.instantiate(cfg)
