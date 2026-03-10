#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — Denoiser Worker (NIXL)

Loads only the Transformer. Receives embeddings from encoder via NIXL RDMA,
runs denoising, registers latents as NIXL-readable for the VAE worker.
"""

import asyncio
import logging
import os
import sys

import torch
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import (  # noqa: E402
    DenoiserRequest, DenoiserResponse,
    NixlTensorReceiver, NixlTensorSender,
)
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEVICE = os.environ.get("DEVICE", "cuda")


class DenoiserStage:
    def __init__(self):
        self.pipe = None
        self.vae_config = None
        self.receiver = None
        self.sender = None

    def load_model(self):
        from diffusers import WanPipeline
        from diffusers.models import WanTransformer3DModel
        from diffusers.schedulers import UniPCMultistepScheduler

        logger.info("Loading transformer from %s", MODEL_PATH)

        # Load only the transformer + scheduler — avoid loading text encoder
        # and VAE onto GPU (saves ~14 GB VRAM and speeds up startup).
        transformer = WanTransformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(DEVICE)
        scheduler = UniPCMultistepScheduler.from_pretrained(
            MODEL_PATH, subfolder="scheduler"
        )

        # Read VAE config without loading weights — only need 5 scalar values
        # for latent normalization in the VAE decode stage.
        from diffusers import AutoencoderKLWan
        vae_config_obj = AutoencoderKLWan.load_config(MODEL_PATH, subfolder="vae")
        self.vae_config = {
            "latents_mean": vae_config_obj["latents_mean"],
            "latents_std": vae_config_obj["latents_std"],
            "z_dim": vae_config_obj["z_dim"],
            "scale_factor_spatial": vae_config_obj.get("scale_factor_spatial", 8),
            "scale_factor_temporal": vae_config_obj.get("scale_factor_temporal", 4),
        }

        self.pipe = WanPipeline(
            tokenizer=None, text_encoder=None, vae=None,
            transformer=transformer, scheduler=scheduler,
        )

        self.receiver = NixlTensorReceiver()
        self.sender = NixlTensorSender()
        logger.info("Denoiser loaded — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)

        # Warmup: run a tiny denoising pass to trigger CUDA kernel compilation.
        logger.info("Running warmup denoise (1 step) …")
        _warmup_h, _warmup_w, _warmup_f = 128, 128, 5
        _dummy_embeds = torch.randn(1, 1, transformer.config.in_channels if hasattr(transformer.config, "in_channels") else 4096, device=DEVICE, dtype=torch.bfloat16)
        try:
            self.pipe(
                prompt_embeds=_dummy_embeds,
                negative_prompt_embeds=None,
                num_inference_steps=1,
                guidance_scale=1.0,
                height=_warmup_h, width=_warmup_w,
                num_frames=_warmup_f,
                output_type="latent",
            )
        except Exception as e:
            logger.warning("Warmup denoise failed (non-critical): %s", e)
        torch.cuda.empty_cache()
        logger.info("Denoiser ready — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    @dynamo_endpoint(DenoiserRequest, DenoiserResponse)
    async def generate(self, request: DenoiserRequest):
        logger.info("Denoising %dx%d, %d frames, %d steps",
                     request.width, request.height, request.num_frames, request.num_inference_steps)

        embeddings, _ = await self.receiver.pull(request.transfer_meta, DEVICE)
        prompt_embeds = embeddings["prompt_embeds"]
        negative_prompt_embeds = embeddings.get("negative_prompt_embeds")

        generator = torch.Generator(device=DEVICE).manual_seed(request.seed)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                height=request.height, width=request.width,
                num_frames=request.num_frames,
                output_type="latent",
            ),
        )
        latents = result.frames

        transfer_meta = await self.sender.prepare(
            {"latents": latents}, extra=self.vae_config,
        )

        logger.info("Denoised — latents %s, yielding NIXL metadata", list(latents.shape))
        yield DenoiserResponse(transfer_meta=transfer_meta, shape=list(latents.shape)).model_dump()


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    ns = runtime.namespace("disagg_diffusion")
    endpoint = ns.component("denoiser").endpoint("generate")

    stage = DenoiserStage()
    await asyncio.get_event_loop().run_in_executor(None, stage.load_model)

    logger.info("Serving: disagg_diffusion.denoiser.generate")
    await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvloop.install()
    asyncio.run(worker())
