import math

import torch
import torch.nn.functional as F
from torch import nn

from .modules.direct_fusion import (
    DirectFusionDecoder,
    ImageAnchorWeight,
    MultiScaleMaskAwareFusion,
    SharedDirectFusionEncoder,
    crop_to_original,
    crop_weight_maps,
    pad_inputs_to_multiple,
)
from .modules.ffc import FFCBlocks
from .modules.nafblock import NAFBlock
from .modules.swin_transformer import SwinTransformerBottleneck


class DirectFusionRestorationNet(nn.Module):
    """Direct multi-observation restoration network."""

    def __init__(
        self,
        in_channels_per_obs=5,
        channels=(32, 64, 128, 256),
        encoder_blocks=(1, 1, 1, 2),
        bottleneck_blocks=4,
        decoder_blocks=(1, 1, 1),
        pad_multiple=8,
        output_mode="residual_anchor",
        residual_scale_init=0.1,
        residual_scale_max=0.2,
        residual_scale_trainable=True,
        concat_masks_to_encoder=True,
        use_is1_input=False,
        use_lq_input=False,
        use_masks_in_fusion=True,
        mask_resize_mode="bilinear",
        use_mask_encoder_for_fusion=False,
        mask_encoder_channels=16,
        fusion_score_hidden_channels=None,
        fusion_score_depth=2,
        fusion_score_kernel_size=3,
        anchor_weight_mode="scale1",
        anchor_score_hidden_channels=32,
        anchor_score_depth=2,
        anchor_score_kernel_size=3,
        block_type="naf",
        bottleneck_type="ffc",
        transformer_window_size=8,
        transformer_heads=8,
        transformer_mlp_ratio=2.0,
    ):
        super().__init__()
        self.pad_multiple = int(pad_multiple)
        self.output_mode = str(output_mode)
        self.residual_scale_max = float(residual_scale_max)
        self.concat_masks_to_encoder = bool(concat_masks_to_encoder)
        self.use_is1_input = bool(use_is1_input)
        self.use_lq_input = bool(use_lq_input)
        self.use_masks_in_fusion = bool(use_masks_in_fusion)
        self.anchor_weight_mode = str(anchor_weight_mode or "scale1").lower()
        expected_in_channels = 3
        if self.use_is1_input:
            expected_in_channels += 3
        if self.use_lq_input:
            expected_in_channels += 3
        if self.concat_masks_to_encoder:
            expected_in_channels += 2
        if int(in_channels_per_obs) != expected_in_channels:
            raise ValueError(
                "in_channels_per_obs must match concat_masks_to_encoder/use_is1_input/use_lq_input: "
                f"expected {expected_in_channels}, got {in_channels_per_obs}"
            )
        self.encoder = SharedDirectFusionEncoder(
            in_channels_per_obs,
            tuple(channels),
            tuple(encoder_blocks),
            block_type=block_type,
        )
        self.fusion = MultiScaleMaskAwareFusion(
            tuple(channels),
            mask_resize_mode=mask_resize_mode,
            use_mask_encoder=use_mask_encoder_for_fusion,
            mask_encoder_channels=mask_encoder_channels,
            score_hidden_channels=fusion_score_hidden_channels,
            score_depth=fusion_score_depth,
            score_kernel_size=fusion_score_kernel_size,
        )
        if self.anchor_weight_mode in {"scale1", "w1"}:
            self.anchor_weight = None
        elif self.anchor_weight_mode in {"image_score", "w0_image", "separate_image"}:
            self.anchor_weight = ImageAnchorWeight(
                hidden_channels=anchor_score_hidden_channels,
                score_depth=anchor_score_depth,
                score_kernel_size=anchor_score_kernel_size,
            )
        else:
            raise ValueError(f"Unsupported anchor_weight_mode: {self.anchor_weight_mode}")
        self.bottleneck = build_direct_fusion_bottleneck(
            int(channels[-1]),
            bottleneck_type=bottleneck_type,
            num_blocks=bottleneck_blocks,
            transformer_window_size=transformer_window_size,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
        )
        self.decoder = DirectFusionDecoder(tuple(channels), decoder_blocks=tuple(decoder_blocks), block_type=block_type)
        scale_logit = residual_scale_to_logit(residual_scale_init, self.residual_scale_max)
        if residual_scale_trainable:
            self.residual_scale_logit = nn.Parameter(torch.tensor(scale_logit, dtype=torch.float32))
        else:
            self.register_buffer("residual_scale_logit", torch.tensor(scale_logit, dtype=torch.float32))

        if self.output_mode not in {"residual_anchor", "direct_rgb_sigmoid"}:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def forward(self, is2, drop, reflection, valid=None, is1=None, lq=None):
        # is2: [B, K, 3, H, W], drop/reflection: [B, K, 1, H, W]
        b, k, _, h, w = is2.shape
        if self.use_is1_input:
            if is1 is None:
                raise ValueError("is1 must be provided when use_is1_input=True")
            if is1.shape != is2.shape:
                raise ValueError(f"is1 shape must match is2 shape: expected {tuple(is2.shape)}, got {tuple(is1.shape)}")
        if self.use_lq_input:
            if lq is None:
                raise ValueError("lq must be provided when use_lq_input=True")
            if lq.shape != is2.shape:
                raise ValueError(f"lq shape must match is2 shape: expected {tuple(is2.shape)}, got {tuple(lq.shape)}")
        if valid is None:
            valid = torch.ones(b, k, dtype=torch.bool, device=is2.device)
        valid = valid.to(device=is2.device, dtype=torch.bool)

        is2, drop, reflection, pad_hw = pad_inputs_to_multiple(
            is2,
            drop,
            reflection,
            multiple=self.pad_multiple,
        )
        if is1 is not None:
            is1 = pad_optional_input(is1, pad_hw)
        if lq is not None:
            lq = pad_optional_input(lq, pad_hw)
        hp, wp = is2.shape[-2:]

        encoder_inputs = [is2]
        if self.use_is1_input:
            encoder_inputs.append(is1)
        if self.use_lq_input:
            encoder_inputs.append(lq)
        if self.concat_masks_to_encoder:
            encoder_inputs.extend([drop, reflection])
        x = torch.cat(encoder_inputs, dim=2)
        encoded = self.encoder(x.reshape(b * k, x.size(2), hp, wp))
        encoded = tuple(feat.reshape(b, k, feat.size(1), feat.size(2), feat.size(3)) for feat in encoded)

        fusion_drop = drop if self.use_masks_in_fusion else torch.zeros_like(drop)
        fusion_reflection = reflection if self.use_masks_in_fusion else torch.zeros_like(reflection)
        fused, weights = self.fusion(encoded, fusion_drop, fusion_reflection, valid)
        f1, f2, f3, f4 = fused
        context = self.bottleneck(f4)
        decoder_output = self.decoder(context, f3, f2, f1)

        if self.output_mode == "direct_rgb_sigmoid":
            residual = torch.zeros_like(decoder_output)
            residual_scale = decoder_output.new_tensor(0.0)
            anchor = torch.zeros_like(decoder_output)
            anchor_weights = weights[0]
            pred = torch.sigmoid(decoder_output)
        else:
            anchor_weights = (
                weights[0]
                if self.anchor_weight is None
                else self.anchor_weight(is2, fusion_drop, fusion_reflection, valid)
            )
            anchor = (anchor_weights * is2).sum(dim=1)
            residual = torch.tanh(decoder_output)
            residual_scale = self.get_residual_scale().to(device=decoder_output.device, dtype=decoder_output.dtype)
            pred = torch.clamp(anchor + residual_scale * residual, 0.0, 1.0)

        pred = crop_to_original(pred, h, w, pad_hw)
        anchor = crop_to_original(anchor, h, w, pad_hw)
        residual = crop_to_original(residual, h, w, pad_hw)

        cropped_weights = crop_weight_maps(weights, h, w, pad_hw)
        cropped_anchor_weights = crop_to_original(anchor_weights, h, w, pad_hw)
        return {
            "pred": pred,
            "anchor": anchor,
            "residual": residual,
            "residual_scale": residual_scale.detach(),
            "weights": cropped_weights[0],
            "anchor_weights": cropped_anchor_weights,
            "scale_weights": cropped_weights,
        }

    def get_residual_scale(self):
        return self.residual_scale_max * torch.sigmoid(self.residual_scale_logit)


def residual_scale_to_logit(initial_scale, max_scale):
    max_scale = float(max_scale)
    if max_scale <= 0:
        raise ValueError("residual_scale_max must be positive")

    ratio = float(initial_scale) / max_scale
    ratio = min(max(ratio, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(ratio / (1.0 - ratio))


def pad_optional_input(tensor, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def build_direct_fusion_bottleneck(
    channels,
    bottleneck_type="ffc",
    num_blocks=4,
    transformer_window_size=8,
    transformer_heads=8,
    transformer_mlp_ratio=2.0,
):
    bottleneck_type = str(bottleneck_type or "ffc").lower()
    if bottleneck_type == "ffc":
        return FFCBlocks(channels, num_blocks=num_blocks)
    if bottleneck_type == "naf":
        return nn.Sequential(*(NAFBlock(channels) for _ in range(int(num_blocks))))
    if bottleneck_type == "transformer":
        return SwinTransformerBottleneck(
            channels,
            num_blocks=num_blocks,
            window_size=transformer_window_size,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
        )
    raise ValueError(f"Unsupported direct fusion bottleneck type: {bottleneck_type}")
