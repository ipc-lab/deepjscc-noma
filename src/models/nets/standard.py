import torch
from torch import nn


class StandardNet(nn.Module):
    def __init__(self, channel, power_constraint, encoder, decoder, num_devices, M):
        super().__init__()

        self.channel = channel
        self.power_constraint = power_constraint
        self.M = M

        self.num_devices = num_devices

        self.encoders = nn.ModuleList([encoder() for _ in range(self.num_devices)])
        self.decoders = nn.ModuleList([decoder() for _ in range(self.num_devices)])

    def forward(self, batch):
        x, csi = batch

        transmissions = []
        for i in range(self.num_devices):
            t = self.encoders[i]((x[:, i, ...], csi))
            transmissions.append(t)

        x = torch.stack(transmissions, dim=1)

        x = self.power_constraint(x)
        x = self.channel((x, csi))

        results = []
        for i in range(self.num_devices):
            t = self.decoders[i]((x, csi))
            results.append(t)

        x = torch.stack(results, dim=1)

        return x
