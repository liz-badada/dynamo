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

        logger.info("Loading transformer from %s", MODEL_PATH)
        self.pipe = WanPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        self.pipe.to(DEVICE)

        self.vae_config = {
            "latents_mean": self.pipe.vae.config.latents_mean,
            "latents_std": self.pipe.vae.config.latents_std,
            "z_dim": self.pipe.vae.config.z_dim,
            "scale_factor_spatial": self.pipe.vae.config.scale_factor_spatial,
            "scale_factor_temporal": self.pipe.vae.config.scale_factor_temporal,
        }
        self.pipe.text_encoder = None
        self.pipe.tokenizer = None
        self.pipe.vae = None
        torch.cuda.empty_cache()

        self.receiver = NixlTensorReceiver()
        self.sender = NixlTensorSender()
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
