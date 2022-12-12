import torch
from torch import nn
from torch.nn import functional as F


class AveragePowerConstraint(nn.Module):
    def __init__(self, power, num_devices):
        super().__init__()

        self.power_factor = torch.sqrt(torch.tensor(power))

        self.num_devices_factor = 1.0 / torch.sqrt(torch.tensor(num_devices))

    def forward(self, hids, mult=1.0):
        hids_shape = hids.size()
        hids = hids.view(hids_shape[0] * hids.shape[1], -1)

        hids = mult * torch.sqrt(1.0/torch.tensor(hids_shape[1], device=hids.device)) * self.power_factor * F.normalize(hids) * torch.sqrt(torch.tensor(hids.size(1), device=hids.device))
        """
        k = 8
        # 2 devices
        0.5 * power * 2 * k
        # 1 device
        1 * power * k
        """

        hids = hids.view(hids_shape)
        

        return hids

class ComplexAveragePowerConstraint(nn.Module):
    
    def __init__(self, power, num_devices):
        super().__init__()
        
        self.power_factor = torch.sqrt(torch.tensor(power))
        self.num_devices_factor = 1.0 / torch.sqrt(torch.tensor(num_devices))

    def forward(self, hids, mult=1.0):
        hids_shape = hids.size()
        hids = hids.view(hids_shape[0] * hids.shape[1], 2, -1)
        
        hids = torch.complex(hids[:, 0, :], hids[:, 1, :])
        
        norm_factor = mult*torch.sqrt(1.0 / torch.tensor(hids_shape[1])) * self.power_factor * torch.sqrt(torch.tensor(hids.real.size(1), device=hids.device))
        """
        # 2 devices
        0.5 * power * k
        # 1 device
        1 * power * 0.5 * k
        """
        
        hids = hids * torch.complex(norm_factor/torch.sqrt(torch.sum((hids * torch.conj(hids)).real, keepdims=True, dim=1)), torch.tensor(0.0, device=hids.device))

        hids = torch.cat([hids.real, hids.imag], dim=1)
        hids = hids.view(hids_shape)

        return hids