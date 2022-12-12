import torch
from torch import nn

from src.models.components.channels import ComplexAWGNMAC


class SingleModelNet(nn.Module):
    def __init__(
        self, channel, power_constraint, encoder, decoder, num_devices, M, ckpt_path=None
    ):
        super().__init__()

        self.channel = channel
        self.power_constraint = power_constraint
        self.M = M

        self.num_devices = num_devices

        self.encoders = nn.ModuleList([encoder(C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([decoder(C=3 * self.num_devices) for _ in range(1)])

        self.device_images = nn.Embedding(
            self.num_devices, embedding_dim=32 * 32
        )  # torch.randn((1, 2, 1, 32, 32), dtype=torch.float32).to("cuda:1")

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)["state_dict"]

            enc_state_dict = {}
            dec_state_dict = {}
            images_state_dict = {}

            for k, v in state_dict.items():
                if k.startswith("net.encoders.0"):
                    enc_state_dict[k.replace("net.encoders.0", "0")] = v
                elif k.startswith("net.decoders.0"):
                    dec_state_dict[k.replace("net.decoders.0", "0")] = v
                elif k.startswith("net.device_images."):
                    images_state_dict[k.replace("net.device_images.", "")] = v

            self.encoders.load_state_dict(enc_state_dict)
            self.decoders.load_state_dict(dec_state_dict)
            self.device_images.load_state_dict(images_state_dict)
            print("checkpoint loaded")

    def forward(self, batch):
        x, csi = batch

        emb = torch.stack(
            [
                self.device_images(
                    torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i
                ).view(x.size(0), 1, 32, 32)
                for i in range(self.num_devices)
            ],
            dim=1,
        )

        x = torch.cat([x, emb], dim=2)

        transmissions = []
        for i in range(self.num_devices):
            t = self.encoders[0]((x[:, i, ...], csi))
            transmissions.append(t)

        x = torch.stack(transmissions, dim=1)

        x = self.power_constraint(x)
        x = self.channel((x, csi))

        x = self.decoders[0]((x, csi))
        x = x.view(x.size(0), self.num_devices, 3, x.size(2), x.size(3))

        return x


class PerfectSICSingleModelNet(nn.Module):
    def __init__(self, channel, power_constraint, encoder, decoder, num_devices, M):
        super().__init__()

        self.channel = channel
        self.power_constraint = power_constraint
        self.M = M

        self.num_devices = num_devices

        self.encoders = nn.ModuleList([encoder(C=4) for _ in range(1)])
        self.decoders = nn.ModuleList([decoder(C=3 * self.num_devices) for _ in range(1)])

        self.mse = nn.MSELoss()

        self.device_images = nn.Embedding(
            self.num_devices, embedding_dim=32 * 32
        )  # torch.randn((1, 2, 1, 32, 32), dtype=torch.float32).to("cuda:1")

    def forward(self, batch):
        x, csi = batch

        emb = torch.stack(
            [
                self.device_images(
                    torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i
                ).view(x.size(0), 1, 32, 32)
                for i in range(self.num_devices)
            ],
            dim=1,
        )
        awgn = None

        x = torch.cat([x, emb], dim=2)

        transmissions = []
        for i in range(self.num_devices):
            t = self.encoders[0]((x[:, i, ...], csi))
            t = self.power_constraint(
                t[:, None, ...], mult=torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))
            )

            t = t.sum(dim=1)

            if awgn is None:
                awgn = torch.randn_like(t) * torch.sqrt(10.0 ** (-csi[..., None, None] / 10.0))

                if isinstance(self.channel, ComplexAWGNMAC):
                    awgn = awgn * torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))

            t = t + awgn

            transmissions.append(t)

        results = []
        for i in range(self.num_devices):
            x = self.decoders[0]((transmissions[i], csi))
            xi = x.view(x.size(0), self.num_devices, 3, x.size(2), x.size(3))[:, i, ...]
            results.append(xi)
        
        x = torch.stack(results, dim=1)

        return x


class MultiEncoderSingleModelNet(nn.Module):
    def __init__(self, channel, power_constraint, encoder, decoder, num_devices, M):
        super().__init__()

        self.channel = channel
        self.power_constraint = power_constraint
        self.M = M

        self.num_devices = num_devices

        self.encoders = nn.ModuleList([encoder(C=3) for _ in range(self.num_devices)])
        self.decoders = nn.ModuleList([decoder(C=6) for _ in range(1)])

    def forward(self, batch):
        x, csi = batch

        transmissions = []
        for i in range(self.num_devices):
            t = self.encoders[i]((x[:, i, ...], csi))
            transmissions.append(t)

        x = torch.stack(transmissions, dim=1)

        x = self.power_constraint(x)
        x = self.channel((x, csi))

        x = self.decoders[0]((x, csi))
        x = x.view(x.size(0), 2, 3, x.size(2), x.size(3))

        return x
