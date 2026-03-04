#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — Encoder Worker (NIXL)

Loads only UMT5 text encoder. Embeddings are registered as NIXL-readable;
only metadata is returned over RPC. The denoiser pulls the actual tensor
data via RDMA.
"""

import asyncio
import logging
import os
import sys

import torch
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import EncoderRequest, EncoderResponse, NixlTensorSender  # noqa: E402
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEVICE = os.environ.get("DEVICE", "cuda")


class EncoderStage:
    def __init__(self):
        self.pipe = None
        self.sender = None

    def load_model(self):
        from transformers import UMT5EncoderModel, T5TokenizerFast
        from diffusers import WanPipeline
        from diffusers.schedulers import UniPCMultistepScheduler

        logger.info("Loading text encoder from %s", MODEL_PATH)
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).to(DEVICE)
        scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
        self.pipe = WanPipeline(
            tokenizer=tokenizer, text_encoder=text_encoder,
            vae=None, transformer=None, scheduler=scheduler,
        )
        self.sender = NixlTensorSender()
        logger.info("Encoder ready — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)

    @dynamo_endpoint(EncoderRequest, EncoderResponse)
    async def generate(self, request: EncoderRequest):
        logger.info("Encoding prompt: %.80s…", request.prompt)
        do_cfg = request.guidance_scale > 1.0

        loop = asyncio.get_event_loop()
        prompt_embeds, negative_prompt_embeds = await loop.run_in_executor(
            None,
            lambda: self.pipe.encode_prompt(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt if do_cfg else None,
                do_classifier_free_guidance=do_cfg,
                device=DEVICE,
            ),
        )

        payload = {"prompt_embeds": prompt_embeds}
        if negative_prompt_embeds is not None:
            payload["negative_prompt_embeds"] = negative_prompt_embeds

        transfer_meta = await self.sender.prepare(payload)

        shapes = {k: list(v.shape) for k, v in payload.items()}
        logger.info("Encoded — shapes %s, yielding NIXL metadata", shapes)

        yield EncoderResponse(transfer_meta=transfer_meta, shapes=shapes).model_dump()


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    ns = runtime.namespace("disagg_diffusion")
    endpoint = ns.component("encoder").endpoint("generate")

    stage = EncoderStage()
    await asyncio.get_event_loop().run_in_executor(None, stage.load_model)

    logger.info("Serving: disagg_diffusion.encoder.generate")
    await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvloop.install()
    asyncio.run(worker())
