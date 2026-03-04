#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — VAE Worker (NIXL)

Loads only AutoencoderKLWan. Receives denoised latents from the denoiser
via NIXL RDMA, decodes to video, returns base64-encoded mp4.
"""

import asyncio
import base64
import io
import logging
import os
import sys
import tempfile

import numpy as np
import torch
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import VAEDecodeRequest, VAEDecodeResponse, NixlTensorReceiver  # noqa: E402
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEVICE = os.environ.get("DEVICE", "cuda")


class VAEStage:
    def __init__(self):
        self.vae = None
        self.receiver = None

    def load_model(self):
        from diffusers import AutoencoderKLWan

        logger.info("Loading VAE from %s", MODEL_PATH)
        self.vae = AutoencoderKLWan.from_pretrained(
            MODEL_PATH, subfolder="vae", torch_dtype=torch.float32
        ).to(DEVICE)
        self.receiver = NixlTensorReceiver()
        logger.info("VAE ready — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    @dynamo_endpoint(VAEDecodeRequest, VAEDecodeResponse)
    async def generate(self, request: VAEDecodeRequest):
        logger.info("Decoding latents via NIXL …")

        tensors, vae_config = await self.receiver.pull(request.transfer_meta, DEVICE)
        latents = tensors["latents"]

        loop = asyncio.get_event_loop()

        def _decode():
            from diffusers.video_processor import VideoProcessor

            lat = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(vae_config["latents_mean"])
                .view(1, vae_config["z_dim"], 1, 1, 1)
                .to(lat.device, lat.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(vae_config["latents_std"])
                .view(1, vae_config["z_dim"], 1, 1, 1)
                .to(lat.device, lat.dtype)
            )
            lat = lat / latents_std + latents_mean
            with torch.no_grad():
                video = self.vae.decode(lat, return_dict=False)[0]
            processor = VideoProcessor(vae_scale_factor=vae_config["scale_factor_spatial"])
            return processor.postprocess_video(video, output_type="np")[0]

        frames = await loop.run_in_executor(None, _decode)

        video_b64 = await loop.run_in_executor(None, _frames_to_mp4_b64, frames)

        logger.info("Decoded — %d frames", len(frames))
        yield VAEDecodeResponse(video_b64=video_b64, num_frames=len(frames)).model_dump()


def _frames_to_mp4_b64(frames) -> str:
    try:
        from diffusers.utils import export_to_video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        export_to_video(frames, tmp_path, fps=16)
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        logger.warning("mp4 failed (%s), encoding first frame as PNG", e)
        from PIL import Image
        frame = frames[0]
        if isinstance(frame, np.ndarray) and frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    ns = runtime.namespace("disagg_diffusion")
    endpoint = ns.component("vae").endpoint("generate")

    stage = VAEStage()
    await asyncio.get_event_loop().run_in_executor(None, stage.load_model)

    logger.info("Serving: disagg_diffusion.vae.generate")
    await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvloop.install()
    asyncio.run(worker())
