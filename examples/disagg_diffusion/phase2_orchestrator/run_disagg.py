#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion Orchestrator — HTTP Server

Persistent server that accepts video generation requests and chains
Encoder → Denoiser → VAE via Dynamo RPC + NIXL RDMA.

Usage:
    python run_disagg.py [--port 8080]

API:
    POST /v1/videos/generations
    GET  /health
    GET  /videos/<filename>

Curl example:
    curl -X POST http://localhost:8080/v1/videos/generations \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "A cat walking on grass", "num_inference_steps": 10}'
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

import uvloop

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1_workers"))

from protocol import DenoiserRequest, EncoderRequest, VAEDecodeRequest  # noqa: E402
from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_videos")


async def call_stage(client, request_json: str) -> dict:
    result = None
    stream = await client.generate(request_json)
    async for chunk in stream:
        data = chunk.data() if hasattr(chunk, "data") else chunk
        if isinstance(data, str):
            data = json.loads(data)
        result = data
    if result is None:
        raise RuntimeError("Empty response from stage")
    return result


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    ns = runtime.namespace("disagg_diffusion")
    encoder_client = await ns.component("encoder").endpoint("generate").client()
    denoiser_client = await ns.component("denoiser").endpoint("generate").client()
    vae_client = await ns.component("vae").endpoint("generate").client()

    logger.info("Waiting for stage workers …")
    await encoder_client.wait_for_instances()
    await denoiser_client.wait_for_instances()
    await vae_client.wait_for_instances()
    logger.info("All 3 stage workers connected")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async def handle_generate(request: dict) -> dict:
        request_id = str(uuid.uuid4())[:8]
        seed = request.get("seed") or int(time.time()) % 1000000
        timings: Dict[str, float] = {}

        t0 = time.monotonic()
        enc_req = EncoderRequest(
            prompt=request["prompt"],
            negative_prompt=request.get("negative_prompt", "Blurry, low quality, distorted"),
            guidance_scale=request.get("guidance_scale", 5.0),
        )
        enc_resp = await call_stage(encoder_client, enc_req.model_dump_json())
        timings["encoder_s"] = round(time.monotonic() - t0, 3)
        logger.info("[%s] Encoder: %.2fs", request_id, timings["encoder_s"])

        t0 = time.monotonic()
        den_req = DenoiserRequest(
            transfer_meta=enc_resp["transfer_meta"],
            height=request.get("height", 480),
            width=request.get("width", 832),
            num_frames=request.get("num_frames", 17),
            num_inference_steps=request.get("num_inference_steps", 20),
            guidance_scale=request.get("guidance_scale", 5.0),
            seed=seed,
        )
        den_resp = await call_stage(denoiser_client, den_req.model_dump_json())
        timings["denoiser_s"] = round(time.monotonic() - t0, 3)
        logger.info("[%s] Denoiser: %.2fs", request_id, timings["denoiser_s"])

        t0 = time.monotonic()
        vae_req = VAEDecodeRequest(transfer_meta=den_resp["transfer_meta"])
        vae_resp = await call_stage(vae_client, vae_req.model_dump_json())
        timings["vae_s"] = round(time.monotonic() - t0, 3)
        timings["total_s"] = round(sum(timings.values()), 3)
        logger.info("[%s] VAE: %.2fs | Total: %.2fs", request_id, timings["vae_s"], timings["total_s"])

        video_b64 = vae_resp["video_b64"]
        resp_format = request.get("response_format", "url")

        if resp_format == "url":
            filename = f"{request_id}.mp4"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(video_b64))
            data = [{"url": f"/videos/{filename}"}]
        else:
            data = [{"b64_json": video_b64}]

        return {
            "id": f"video-{request_id}",
            "created": int(time.time()),
            "data": data,
            "timings": timings,
        }

    # --- HTTP server using aiohttp (lightweight, runs in same event loop) ---
    from aiohttp import web

    async def handle_post(http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
            result = await handle_generate(body)
            return web.json_response(result)
        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(http_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_video(http_request: web.Request) -> web.Response:
        filename = http_request.match_info["filename"]
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(filepath, headers={"Content-Type": "video/mp4"})

    http_app = web.Application()
    http_app.router.add_post("/v1/videos/generations", handle_post)
    http_app.router.add_get("/health", handle_health)
    http_app.router.add_get("/videos/{filename}", handle_video)

    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()

    logger.info("Server listening on http://%s:%d", HOST, PORT)
    logger.info("  POST /v1/videos/generations")
    logger.info("  GET  /health")
    logger.info("  GET  /videos/<id>.mp4")

    # Keep alive forever
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvloop.install()
    asyncio.run(worker())
