#!/usr/bin/env python
"""Minimal FUMO stage2 runtime: LQ + P_int -> diffusion prelim -> refine final."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer, CLIPTextModel

from basicsr.models.archs.NAFNet_arch import NAFNet
from diffusion.controlnetvae import ControlNetVAEModel
from diffusion.pipeline_onestep import OneStepPipeline


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    channels = image.shape[1]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)[None, None]
    kernel = kernel.repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(image, kernel, groups=channels, dilation=radius)


def wavelet_decomposition(image: Tensor, levels: int = 5) -> tuple[Tensor, Tensor]:
    high_freq = torch.zeros_like(image)
    low_freq = image
    for level in range(levels):
        radius = 2**level
        low_freq = wavelet_blur(image, radius)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def normalize_to_01(x: Tensor, eps: float = 1e-6) -> Tensor:
    dims = (1, 2, 3)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def compute_hf_image(image: Tensor) -> Tensor:
    high_freq, _ = wavelet_decomposition(image)
    return normalize_to_01(high_freq)


def compute_hf_mag(image: Tensor) -> Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
    mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return (hf_mag / mean).clamp(0.0, 1.0)


def load_prior_tensor(prior_path: str | Path) -> Tensor:
    prior = np.load(prior_path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    prior_img = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
    return TF.to_tensor(prior_img).unsqueeze(0)


def load_pipeline(args: SimpleNamespace, device: torch.device, dtype: torch.dtype):
    controlnet = ControlNetVAEModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet_dir, torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)

    pipe = OneStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(device)
    pipe.default_processing_resolution = int(args.resolution)
    pipe.set_progress_bar_config(disable=True)

    for module in (controlnet, unet, vae, text_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return pipe


def infer_diff_prelim(pipeline, image_tensor: Tensor, prior_tensor: Tensor, prompt: str, beta: float) -> Tensor:
    device = pipeline._execution_device
    dtype = pipeline.dtype

    if pipeline.empty_text_embedding is None:
        text_inputs = pipeline.tokenizer(
            "",
            padding="do_not_pad",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pipeline.empty_text_embedding = pipeline.text_encoder(text_inputs.input_ids.to(device))[0]

    if pipeline.prompt_embeds is None or pipeline.prompt != prompt:
        pipeline.prompt = prompt
        pipeline.prompt_embeds = None

    if pipeline.prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            pipeline.prompt,
            device,
            1,
            False,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        pipeline.prompt_embeds = prompt_embeds
        pipeline.negative_prompt_embeds = negative_prompt_embeds

    image, padding, original_resolution = pipeline.image_processor.preprocess(
        image_tensor, pipeline.default_processing_resolution, "bilinear", device, dtype
    )
    if pipeline.prompt_embeds.shape[0] != image.shape[0]:
        pipeline.prompt_embeds = pipeline.prompt_embeds.repeat(image.shape[0], 1, 1)
    image_latent, pred_latent = pipeline.prepare_latents(image, None, None, 1, 1)

    prior = prior_tensor.to(device=device, dtype=dtype)
    prior = F.interpolate(prior, size=image.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    hf_mag = compute_hf_mag(image)

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        image_latent.detach(),
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        conditioning_scale=1.0,
        guess_mode=False,
        return_dict=False,
    )

    gated_down = []
    for residual in down_block_res_samples:
        prior_res = F.interpolate(prior, size=residual.shape[-2:], mode="area").clamp(0.0, 1.0)
        hf_res = F.interpolate(hf_mag, size=residual.shape[-2:], mode="area").clamp(0.0, 1.0)
        gate = (1.0 + beta * prior_res * hf_res).clamp(1.0, 1.0 + beta)
        gated_down.append(residual * gate)

    prior_mid = F.interpolate(prior, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    hf_mid = F.interpolate(hf_mag, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    gate_mid = (1.0 + beta * prior_mid * hf_mid).clamp(1.0, 1.0 + beta)

    latent_x_t = pipeline.unet(
        pred_latent,
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        down_block_additional_residuals=gated_down,
        mid_block_additional_residual=mid_block_res_sample * gate_mid,
        return_dict=False,
    )[0]

    prediction = pipeline.decode_prediction(latent_x_t)
    prediction = pipeline.image_processor.unpad_image(prediction, padding)
    return pipeline.image_processor.resize_antialias(prediction, original_resolution, "bilinear", is_aa=False)


def build_refine_input(prelim: Tensor, cond: Tensor, prior: Tensor) -> Tensor:
    return torch.cat([prelim, compute_hf_image(cond), prior, cond], dim=1)


def load_refine_models(args: SimpleNamespace, device: torch.device):
    in_ch = 10
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blk_num,
        enc_blk_nums=args.nafnet_enc_blk_nums,
        dec_blk_nums=args.nafnet_dec_blk_nums,
    ).to(device)
    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)
    refine_net.load_state_dict(torch.load(args.refine_net_path, map_location="cpu"))
    refine_head.load_state_dict(torch.load(args.refine_head_path, map_location="cpu"))
    refine_net.eval()
    refine_head.eval()
    return refine_net, refine_head


def apply_refine_residual(refine_net, refine_head, prelim: Tensor, cond: Tensor, prior: Tensor, residual_scale: float) -> Tensor:
    feat = refine_net(build_refine_input(prelim, cond, prior))
    residual = torch.tanh(refine_head(feat)) * residual_scale
    return (prelim + residual).clamp(0.0, 1.0)
