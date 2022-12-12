from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)
import torch.nn as nn

from .afmodule import AFModule

class DeepJSCCQ2Encoder(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=C,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=M),
            AFModule(N=M, num_dim=1),
            AttentionBlock(M),
        ])
    

    def forward(self, x):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x


class DeepJSCCQ2Decoder(nn.Module):

    def __init__(self, N, M, C=3, **kwargs):
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(M),
            ResidualBlock(
                in_ch=M,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(N=N, num_dim=1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            AFModule(N=N, num_dim=1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=C,
                upsample=2),
            AFModule(N=C, num_dim=1),
        ])


    def forward(self, x):

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x
