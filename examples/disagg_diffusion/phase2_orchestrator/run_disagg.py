#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion Orchestrator — HTTP Server with Pipeline Parallelism

Persistent server that accepts video generation requests and chains
Encoder → Denoiser → VAE via Dynamo RPC + NIXL RDMA.

Pipeline parallelism: multiple requests can be in different stages
simultaneously. Per-stage semaphores control backpressure so each
GPU processes one request at a time, while the pipeline stays full.

    Request 1:  [Encoder] → [Denoiser] → [  VAE  ]
    Request 2:             [Encoder] → [Denoiser] → [  VAE  ]
    Request 3:                        [Encoder] → [Denoiser] → ...

Usage:
    python run_disagg.py [--port 8080]

API:
    POST /v1/videos/generations
    GET  /health
    GET  /pipeline/status
    GET  /videos/<filename>
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from typing import Dict

import uvloop

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1_workers"))

from protocol import DenoiserRequest, EncoderRequest, VAEDecodeRequest  # noqa: E402
from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_videos")
# Max requests allowed in the pipeline at once (prevents OOM from queuing).
MAX_PIPELINE_DEPTH = int(os.environ.get("MAX_PIPELINE_DEPTH", "4"))


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


class PipelineTracker:
    """Tracks requests flowing through the 3-stage pipeline."""

    STAGES = ("encoder", "denoiser", "vae")

    def __init__(self):
        self._active: Dict[str, str] = {}   # request_id → current stage
        self._completed = 0
        self._failed = 0
        self._stage_times: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def enter(self, request_id: str, stage: str):
        async with self._lock:
            self._active[request_id] = stage

    async def leave(self, request_id: str, stage: str, elapsed: float):
        async with self._lock:
            self._stage_times[stage].append(elapsed)
            if request_id in self._active and self._active[request_id] == stage:
                if stage == "vae":
                    del self._active[request_id]

    async def mark_done(self, request_id: str):
        async with self._lock:
            self._active.pop(request_id, None)
            self._completed += 1

    async def mark_failed(self, request_id: str):
        async with self._lock:
            self._active.pop(request_id, None)
            self._failed += 1

    async def status(self) -> dict:
        async with self._lock:
            per_stage = defaultdict(list)
            for rid, stage in self._active.items():
                per_stage[stage].append(rid)
            avg_times = {}
            for stage in self.STAGES:
                times = self._stage_times[stage]
                avg_times[stage] = round(sum(times) / len(times), 3) if times else 0
            return {
                "active_requests": dict(per_stage),
                "active_count": len(self._active),
                "completed": self._completed,
                "failed": self._failed,
                "avg_stage_seconds": avg_times,
            }


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

    # --- Pipeline parallelism primitives ---
    # Per-stage semaphores: each stage has 1 GPU, so allow 1 concurrent
    # compute per stage. Requests queue at the semaphore boundary,
    # creating natural pipeline overlap.
    stage_sems = {
        "encoder": asyncio.Semaphore(1),
        "denoiser": asyncio.Semaphore(1),
        "vae": asyncio.Semaphore(1),
    }
    # Overall admission control to prevent unbounded queuing.
    admission = asyncio.Semaphore(MAX_PIPELINE_DEPTH)
    tracker = PipelineTracker()

    async def run_stage(name: str, request_id: str, client, request_json: str) -> dict:
        """Run a single stage with semaphore gating and tracking."""
        async with stage_sems[name]:
            await tracker.enter(request_id, name)
            t0 = time.monotonic()
            result = await call_stage(client, request_json)
            elapsed = time.monotonic() - t0
            await tracker.leave(request_id, name, elapsed)
            logger.info("[%s] %s: %.2fs", request_id, name.capitalize(), elapsed)
            return result, elapsed

    async def handle_generate(request: dict) -> dict:
        request_id = str(uuid.uuid4())[:8]
        seed = request.get("seed") or int(time.time()) % 1000000
        timings: Dict[str, float] = {}

        async with admission:
            # Stage 1: Encoder
            enc_req = EncoderRequest(
                prompt=request["prompt"],
                negative_prompt=request.get("negative_prompt", "Blurry, low quality, distorted"),
                guidance_scale=request.get("guidance_scale", 5.0),
            )
            enc_resp, timings["encoder_s"] = await run_stage(
                "encoder", request_id, encoder_client, enc_req.model_dump_json(),
            )

            # Stage 2: Denoiser
            den_req = DenoiserRequest(
                transfer_meta=enc_resp["transfer_meta"],
                height=request.get("height", 480),
                width=request.get("width", 832),
                num_frames=request.get("num_frames", 17),
                num_inference_steps=request.get("num_inference_steps", 20),
                guidance_scale=request.get("guidance_scale", 5.0),
                seed=seed,
            )
            den_resp, timings["denoiser_s"] = await run_stage(
                "denoiser", request_id, denoiser_client, den_req.model_dump_json(),
            )

            # Stage 3: VAE — writes mp4 to shared storage, returns filename.
            vae_req = VAEDecodeRequest(
                transfer_meta=den_resp["transfer_meta"],
                request_id=request_id,
            )
            vae_resp, timings["vae_s"] = await run_stage(
                "vae", request_id, vae_client, vae_req.model_dump_json(),
            )

        timings["total_s"] = round(sum(timings.values()), 3)
        await tracker.mark_done(request_id)
        logger.info("[%s] Total: %.2fs", request_id, timings["total_s"])

        # VAE wrote the file directly — just reference it.
        filename = vae_resp["video_path"]
        resp_format = request.get("response_format", "url")

        if resp_format == "url":
            data = [{"url": f"/videos/{filename}"}]
        else:
            # Fallback: read the file and base64 encode on-demand.
            import base64
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "rb") as f:
                data = [{"b64_json": base64.b64encode(f.read()).decode("ascii")}]

        return {
            "id": f"video-{request_id}",
            "created": int(time.time()),
            "data": data,
            "timings": timings,
        }

    # --- HTTP server ---
    from aiohttp import web

    async def handle_post(http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
            if "prompt" not in body:
                return web.json_response({"error": "missing 'prompt' field"}, status=400)
            result = await handle_generate(body)
            return web.json_response(result)
        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(http_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_pipeline_status(http_request: web.Request) -> web.Response:
        status = await tracker.status()
        return web.json_response(status)

    async def handle_video(http_request: web.Request) -> web.Response:
        filename = http_request.match_info["filename"]
        # Prevent path traversal.
        if "/" in filename or "\\" in filename or ".." in filename:
            return web.json_response({"error": "invalid filename"}, status=400)
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(filepath, headers={"Content-Type": "video/mp4"})

    http_app = web.Application()
    http_app.router.add_post("/v1/videos/generations", handle_post)
    http_app.router.add_get("/health", handle_health)
    http_app.router.add_get("/pipeline/status", handle_pipeline_status)
    http_app.router.add_get("/videos/{filename}", handle_video)

    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()

    logger.info("Server listening on http://%s:%d (pipeline depth=%d)", HOST, PORT, MAX_PIPELINE_DEPTH)
    logger.info("  POST /v1/videos/generations")
    logger.info("  GET  /health")
    logger.info("  GET  /pipeline/status")
    logger.info("  GET  /videos/<id>.mp4")

    # Keep alive forever
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvloop.install()
    asyncio.run(worker())
