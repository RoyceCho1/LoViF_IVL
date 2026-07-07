import torch
from torch import nn
import torch.nn.functional as F


class SwinTransformerBottleneck(nn.Module):
    """Swin-style window transformer bottleneck for low-resolution features."""

    def __init__(self, channels, num_blocks=4, window_size=8, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        window_size = int(window_size)
        self.blocks = nn.Sequential(
            *(
                SwinTransformerBlock(
                    channels,
                    window_size=window_size,
                    shift_size=0 if block_index % 2 == 0 else window_size // 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for block_index in range(int(num_blocks))
            )
        )

    def forward(self, x):
        return self.blocks(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, channels, window_size=8, shift_size=0, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        channels = int(channels)
        window_size = int(window_size)
        shift_size = int(shift_size)
        num_heads = int(num_heads)
        if channels % num_heads != 0:
            raise ValueError(f"channels must be divisible by num_heads: {channels} % {num_heads}")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError("shift_size must satisfy 0 <= shift_size < window_size")

        hidden_channels = int(round(channels * float(mlp_ratio)))
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x, padded_hw = pad_to_window(x, self.window_size)
        hp, wp = padded_hw
        shift_size = self.shift_size if min(hp, wp) > self.window_size else 0

        if shift_size > 0:
            shifted = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(-2, -1))
            attn_mask = build_shifted_window_mask(
                hp,
                wp,
                self.window_size,
                shift_size,
                device=x.device,
            )
        else:
            shifted = x
            attn_mask = None

        windows = partition_windows(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        windows = windows + self.attn(self.norm1(windows), mask=attn_mask)
        windows = windows + self.mlp(self.norm2(windows))

        shifted = windows.view(-1, self.window_size, self.window_size, c)
        x = reverse_windows(shifted, self.window_size, hp, wp, b)
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(-2, -1))
        return x[..., :h, :w]


class WindowAttention(nn.Module):
    """Window multi-head self-attention with relative position bias."""

    def __init__(self, channels, window_size=8, num_heads=8):
        super().__init__()
        channels = int(channels)
        window_size = int(window_size)
        num_heads = int(num_heads)
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        self.register_buffer(
            "relative_position_index",
            build_relative_position_index(window_size),
            persistent=False,
        )
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.size(0)
            batch_size = batch_windows // num_windows
            attn = attn.view(batch_size, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = torch.softmax(attn, dim=-1)
        output = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(output)


def pad_to_window(x, window_size):
    _, _, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x, x.shape[-2:]


def partition_windows(x, window_size):
    batch_size, channels, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
    )
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size, window_size, channels)


def reverse_windows(windows, window_size, height, width, batch_size):
    channels = windows.size(-1)
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(batch_size, channels, height, width)


def build_relative_position_index(window_size):
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij",
        )
    )
    coords_flatten = coords.flatten(1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1)


def build_shifted_window_mask(height, width, window_size, shift_size, device):
    img_mask = torch.zeros((1, 1, height, width), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, :, h_slice, w_slice] = count
            count += 1

    mask_windows = partition_windows(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
    return attn_mask.masked_fill(attn_mask == 0, 0.0)
