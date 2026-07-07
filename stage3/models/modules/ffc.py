from contextlib import nullcontext

import torch
import torch.nn as nn


class FourierUnit(nn.Module):
    """Frequency-domain 1x1 convolution over real/imaginary FFT components."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels * 2, 1)
        self.act = nn.GELU()

    def forward(self, x):
        dtype = x.dtype
        autocast_context = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else nullcontext()
        with autocast_context:
            x_float = x.float()
            ffted = torch.fft.rfft2(x_float, norm="ortho")
            spectral = torch.cat([ffted.real, ffted.imag], dim=1)
            spectral = self.act(self.conv(spectral))
            real, imag = spectral.chunk(2, dim=1)
            ffted = torch.complex(real, imag)
            output = torch.fft.irfft2(ffted, s=x_float.shape[-2:], norm="ortho")
        return output.to(dtype=dtype)


class FastFourierConvolution(nn.Module):
    """Local spatial convolution plus global Fourier convolution."""

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.global_unit = FourierUnit(channels, channels)
        self.mix = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        local = self.local(x)
        global_feature = self.global_unit(x)
        return self.mix(torch.cat([local, global_feature], dim=1))


class FFCBlock(nn.Module):
    """Residual Fast Fourier Convolution block for bottleneck context modeling."""

    def __init__(self, channels):
        super().__init__()
        self.ffc = FastFourierConvolution(channels)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        return x + self.ffc(x) * self.scale


class FFCBlocks(nn.Module):
    def __init__(self, channels, num_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(*(FFCBlock(channels) for _ in range(num_blocks)))

    def forward(self, x):
        return self.blocks(x)
