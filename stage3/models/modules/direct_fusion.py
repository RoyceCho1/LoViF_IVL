import torch
from torch import nn
import torch.nn.functional as F

from .ffc import FFCBlock
from .nafblock import NAFBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, block_type="naf"):
        super().__init__()
        block_type = str(block_type or "naf").lower()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
        ]
        layers.extend(build_feature_block(out_channels, block_type) for _ in range(num_blocks))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_feature_block(channels, block_type="naf"):
    block_type = str(block_type or "naf").lower()
    if block_type == "naf":
        return NAFBlock(channels)
    if block_type == "ffc":
        return FFCBlock(channels)
    raise ValueError(f"Unsupported feature block type: {block_type}")


class SharedDirectFusionEncoder(nn.Module):
    """Shared per-observation encoder returning H, H/2, H/4, H/8 features."""

    def __init__(
        self,
        in_channels=5,
        channels=(32, 64, 128, 256),
        blocks_per_scale=(1, 1, 1, 2),
        block_type="naf",
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain four scale widths")
        if len(blocks_per_scale) != 4:
            raise ValueError("blocks_per_scale must contain four values")

        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(in_channels, c1, blocks_per_scale[0], block_type=block_type)
        self.down12 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU())
        self.enc2 = ConvBlock(c2, c2, blocks_per_scale[1], block_type=block_type)
        self.down23 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GELU())
        self.enc3 = ConvBlock(c3, c3, blocks_per_scale[2], block_type=block_type)
        self.down34 = nn.Sequential(nn.Conv2d(c3, c4, 3, stride=2, padding=1), nn.GELU())
        self.enc4 = ConvBlock(c4, c4, blocks_per_scale[3], block_type=block_type)

    def forward(self, x):
        # x: [B*K, 5, H, W]
        f1 = self.enc1(x)
        f2 = self.enc2(self.down12(f1))
        f3 = self.enc3(self.down23(f2))
        f4 = self.enc4(self.down34(f3))
        return f1, f2, f3, f4


class MaskAwareFusion(nn.Module):
    """Fuse observation features with mask-conditioned softmax weights."""

    def __init__(self, channels, hidden_channels=None, score_depth=2, score_kernel_size=3):
        super().__init__()
        hidden_channels = hidden_channels or max(16, channels // 2)
        self.score = build_score_cnn(
            channels + 3,
            hidden_channels,
            depth=score_depth,
            kernel_size=score_kernel_size,
        )

    def forward(self, feats, drop_mask, refl_mask, valid):
        # feats: [B, K, C, Hs, Ws], masks: [B, K, 1, Hs, Ws], valid: [B, K]
        b, k, c, h, w = feats.shape
        valid = valid.to(device=feats.device, dtype=torch.bool)
        deg_mask = torch.clamp(drop_mask + refl_mask, 0.0, 1.0)
        reliability = 1.0 - deg_mask
        gate_input = torch.cat([feats, drop_mask, refl_mask, reliability], dim=2)
        scores = self.score(gate_input.reshape(b * k, c + 3, h, w)).reshape(b, k, 1, h, w)
        scores = scores.masked_fill(~valid[:, :, None, None, None], -1.0e4)
        weights = torch.softmax(scores, dim=1)
        fused = (weights * feats).sum(dim=1)
        return fused, weights


class ImageAnchorWeight(nn.Module):
    """Build full-resolution anchor weights from RGB observations and masks."""

    def __init__(self, hidden_channels=32, score_depth=2, score_kernel_size=3):
        super().__init__()
        self.score = build_score_cnn(
            6,
            hidden_channels,
            depth=score_depth,
            kernel_size=score_kernel_size,
        )

    def forward(self, is2, drop_mask, refl_mask, valid):
        # is2: [B, K, 3, H, W], masks: [B, K, 1, H, W], valid: [B, K]
        b, k, _, h, w = is2.shape
        valid = valid.to(device=is2.device, dtype=torch.bool)
        deg_mask = torch.clamp(drop_mask + refl_mask, 0.0, 1.0)
        reliability = 1.0 - deg_mask
        score_input = torch.cat([is2, drop_mask, refl_mask, reliability], dim=2)
        scores = self.score(score_input.reshape(b * k, 6, h, w)).reshape(b, k, 1, h, w)
        scores = scores.masked_fill(~valid[:, :, None, None, None], -1.0e4)
        return torch.softmax(scores, dim=1)


def build_score_cnn(in_channels, hidden_channels, depth=2, kernel_size=3):
    depth = int(depth)
    if depth < 1:
        raise ValueError("score_depth must be >= 1")
    kernel_size = int(kernel_size)
    padding = kernel_size // 2
    layers = []
    current_channels = int(in_channels)
    hidden_channels = int(hidden_channels)
    for _ in range(depth):
        layers.extend(
            [
                nn.Conv2d(current_channels, hidden_channels, kernel_size, padding=padding),
                nn.GELU(),
            ]
        )
        current_channels = hidden_channels
    layers.append(nn.Conv2d(current_channels, 1, 1))
    return nn.Sequential(*layers)


class ObservationMaskPyramidEncoder(nn.Module):
    """Learn a multi-scale mask pyramid instead of interpolating masks directly."""

    def __init__(self, hidden_channels=16):
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1),
            nn.GELU(),
            NAFBlock(hidden_channels),
        )
        self.down12 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.GELU())
        self.enc2 = NAFBlock(hidden_channels)
        self.down23 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.GELU())
        self.enc3 = NAFBlock(hidden_channels)
        self.down34 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1), nn.GELU())
        self.enc4 = NAFBlock(hidden_channels)
        self.heads = nn.ModuleList(nn.Conv2d(hidden_channels, 2, 3, padding=1) for _ in range(4))

    def forward(self, drop, reflection):
        b, k, _, h, w = drop.shape
        x = torch.cat([drop, reflection], dim=2).reshape(b * k, 2, h, w)
        f1 = self.stem(x)
        f2 = self.enc2(self.down12(f1))
        f3 = self.enc3(self.down23(f2))
        f4 = self.enc4(self.down34(f3))

        levels = []
        for feat, head in zip((f1, f2, f3, f4), self.heads):
            mask_pair = torch.sigmoid(head(feat))
            mask_pair = mask_pair.reshape(b, k, 2, mask_pair.size(-2), mask_pair.size(-1))
            levels.append((mask_pair[:, :, 0:1], mask_pair[:, :, 1:2]))
        return tuple(levels)


class MultiScaleMaskAwareFusion(nn.Module):
    def __init__(
        self,
        channels,
        mask_resize_mode="bilinear",
        use_mask_encoder=False,
        mask_encoder_channels=16,
        score_hidden_channels=None,
        score_depth=2,
        score_kernel_size=3,
    ):
        super().__init__()
        hidden_values = resolve_per_scale_value(score_hidden_channels, len(channels))
        depth_values = resolve_per_scale_value(score_depth, len(channels))
        self.fusions = nn.ModuleList(
            MaskAwareFusion(
                channel,
                hidden_channels=hidden_values[index],
                score_depth=depth_values[index],
                score_kernel_size=score_kernel_size,
            )
            for index, channel in enumerate(channels)
        )
        self.mask_resize_mode = str(mask_resize_mode)
        self.mask_encoder = (
            ObservationMaskPyramidEncoder(mask_encoder_channels)
            if use_mask_encoder
            else None
        )

    def forward(self, features, drop, reflection, valid):
        # features: tuple of four [B, K, C_s, H_s, W_s] tensors.
        fused_features = []
        weight_maps = []
        mask_levels = self.mask_encoder(drop, reflection) if self.mask_encoder is not None else None
        for level_index, (feat, fusion) in enumerate(zip(features, self.fusions)):
            mask_size = feat.shape[-2:]
            if mask_levels is None:
                drop_s = resize_observation_masks(drop, mask_size, mode=self.mask_resize_mode)
                refl_s = resize_observation_masks(reflection, mask_size, mode=self.mask_resize_mode)
            else:
                drop_s, refl_s = mask_levels[level_index]
                if drop_s.shape[-2:] != mask_size:
                    drop_s = resize_observation_masks(drop_s, mask_size, mode=self.mask_resize_mode)
                    refl_s = resize_observation_masks(refl_s, mask_size, mode=self.mask_resize_mode)
            fused, weights = fusion(feat, drop_s, refl_s, valid)
            fused_features.append(fused)
            weight_maps.append(weights)
        return tuple(fused_features), tuple(weight_maps)


def resolve_per_scale_value(value, num_scales):
    if isinstance(value, (list, tuple)):
        if len(value) != num_scales:
            raise ValueError(f"Expected {num_scales} per-scale values, got {len(value)}")
        return list(value)
    return [value for _ in range(num_scales)]


class DirectFusionDecoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), decoder_blocks=(1, 1, 1), block_type="naf"):
        super().__init__()
        if len(decoder_blocks) != 3:
            raise ValueError("decoder_blocks must contain three values")
        c1, c2, c3, c4 = channels
        self.dec3 = ConvBlock(c4 + c3, c3, num_blocks=decoder_blocks[0], block_type=block_type)
        self.dec2 = ConvBlock(c3 + c2, c2, num_blocks=decoder_blocks[1], block_type=block_type)
        self.dec1 = ConvBlock(c2 + c1, c1, num_blocks=decoder_blocks[2], block_type=block_type)
        self.out = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, 3, 3, padding=1),
        )

    def forward(self, context, f3, f2, f1):
        x = F.interpolate(context, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, f3], dim=1))
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, f1], dim=1))
        return self.out(x)


def resize_observation_masks(mask, size, mode="bilinear"):
    b, k, c, h, w = mask.shape
    kwargs = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(mask.reshape(b * k, c, h, w), size=size, mode=mode, **kwargs)
    return resized.reshape(b, k, c, size[0], size[1]).clamp(0.0, 1.0)


def pad_inputs_to_multiple(is2, drop, reflection, multiple=8):
    if multiple <= 1:
        return is2, drop, reflection, (0, 0)
    h, w = is2.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return is2, drop, reflection, (0, 0)
    padding = (0, pad_w, 0, pad_h)
    return (
        F.pad(is2, padding, mode="constant", value=0.0),
        F.pad(drop, padding, mode="constant", value=0.0),
        F.pad(reflection, padding, mode="constant", value=0.0),
        (pad_h, pad_w),
    )


def crop_to_original(tensor, height, width, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return tensor
    return tensor[..., :height, :width]


def crop_weight_maps(weights, height, width, pad_hw):
    cropped = []
    for weight in weights:
        if weight.shape[-2:] == (height, width):
            cropped.append(weight)
        elif weight.shape[-2] >= height and weight.shape[-1] >= width:
            cropped.append(crop_to_original(weight, height, width, pad_hw))
        else:
            cropped.append(weight)
    return tuple(cropped)
