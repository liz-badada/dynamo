#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — VAE Worker (NIXL)

Loads only AutoencoderKLWan. Receives denoised latents from the denoiser
via NIXL RDMA, decodes to video, writes mp4 to shared storage.
Only the file path is returned over RPC (not the video bytes).
"""

import asyncio
import logging
import os
import sys
import uuid

import torch
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import VAEDecodeRequest, VAEDecodeResponse, NixlTensorReceiver  # noqa: E402
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEVICE = os.environ.get("DEVICE", "cuda")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_videos")


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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("VAE loaded — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)

        # Warmup: run a tiny decode to trigger CUDA kernel compilation.
        logger.info("Running warmup decode …")
        _z_dim = self.vae.config.z_dim
        _dummy = torch.randn(1, _z_dim, 2, 4, 4, device=DEVICE, dtype=self.vae.dtype)
        with torch.no_grad():
            self.vae.decode(_dummy, return_dict=False)
        del _dummy
        torch.cuda.empty_cache()
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

        # Write mp4 directly to shared storage — no base64 over RPC.
        request_id = request.request_id or str(uuid.uuid4())[:8]
        filename = f"{request_id}.mp4"
        filepath = os.path.join(OUTPUT_DIR, filename)
        await loop.run_in_executor(None, _save_video, frames, filepath)

        logger.info("Decoded — %d frames → %s", len(frames), filepath)
        yield VAEDecodeResponse(video_path=filename, num_frames=len(frames)).model_dump()


def _save_video(frames, filepath: str):
    """Save frames as mp4. Falls back to PNG if ffmpeg is unavailable."""
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, filepath, fps=16)
    except Exception as e:
        logger.warning("mp4 export failed (%s), saving first frame as PNG", e)
        import numpy as np
        from PIL import Image
        frame = frames[0]
        if isinstance(frame, np.ndarray) and frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        png_path = filepath.replace(".mp4", ".png")
        img.save(png_path)


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
