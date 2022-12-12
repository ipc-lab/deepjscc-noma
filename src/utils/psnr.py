import torch
from torchmetrics import Metric


class PSNR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == targets.shape

        self.psnr += torch.sum(
            -10 * torch.log10(torch.mean((preds - targets) ** 2, dim=[-1, -2, -3]))
        ).to(self.device)
        self.items += preds.shape[0]

    def compute(self):
        return self.psnr / self.items
