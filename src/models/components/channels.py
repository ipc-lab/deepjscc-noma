import torch
from torch import nn
from torch.nn import functional as F


class RealAWGNMAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr = batch

        # inputs: BxCxWxH
        # snr: Bx1

        x = torch.sum(x, 1)

        awgn = torch.randn_like(x) * torch.sqrt(10.0 ** (-snr[..., None, None] / 10.0))

        x = x + awgn

        return x

class ComplexAWGNMAC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr = batch

        # inputs: BxCxWxH
        # snr: Bx1

        x = torch.sum(x, 1)

        awgn = torch.randn_like(x) * torch.sqrt(10.0 ** (-snr[..., None, None] / 10.0))
        
        awgn = awgn * torch.sqrt(torch.tensor(0.5, device=x.device))

        x = x + awgn

        return x


class PerfectChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x, snr = batch

        x = torch.sum(x, 1)

        return x
