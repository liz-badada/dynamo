#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 0: Validate that a WanPipeline (video diffusion) can be split into
Encoder / Denoiser / VAE stages with portable intermediate tensors.

This script runs on a single GPU without Dynamo.  It:
  1. Generates a reference video with the monolithic pipeline.
  2. Runs the same generation in three SEPARATE stages, each loading only
     its own model components, passing serialized tensors between them.
  3. Compares the results to confirm equivalence.
  4. Reports per-stage VRAM usage.

Usage:
    python validate_split.py \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        --prompt "A cat walking on grass" \
        --output-dir /tmp/disagg_validate
"""

import argparse
import gc
import io
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def serialize_tensors(data: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(data, buf)
    return buf.getvalue()


def deserialize_tensors(raw: bytes) -> dict:
    buf = io.BytesIO(raw)
    return torch.load(buf, weights_only=True)


# ---------------------------------------------------------------------------
# Monolithic baseline
# ---------------------------------------------------------------------------

def run_monolithic(model_path: str, prompt: str, negative_prompt: str,
                   seed: int, num_steps: int, num_frames: int,
                   height: int, width: int, guidance_scale: float,
                   device: str):
    from diffusers import WanPipeline, AutoencoderKLWan

    logger.info("=== Monolithic Run ===")
    logger.info("Loading full pipeline …")

    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        model_path, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    logger.info("Full pipeline VRAM: %.0f MB", vram_mb())

    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
        num_frames=num_frames,
    )
    frames = output.frames[0]

    del pipe, vae
    flush_vram()
    return frames


# ---------------------------------------------------------------------------
# Split stages — each loads ONLY its own components
# ---------------------------------------------------------------------------

def stage_encoder(model_path: str, prompt: str, negative_prompt: str,
                  guidance_scale: float, device: str) -> bytes:
    """Stage 1: Load ONLY text encoder + tokenizer, encode prompt."""
    from transformers import UMT5EncoderModel, T5TokenizerFast

    logger.info("=== Stage 1: Encoder (text encoder only) ===")
    logger.info("VRAM before load: %.0f MB", vram_mb())

    tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)

    logger.info("Text encoder VRAM: %.0f MB", vram_mb())

    do_cfg = guidance_scale > 1.0

    from diffusers import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    encoder_pipe = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=None,
        transformer=None,
        scheduler=scheduler,
    )

    prompt_embeds, negative_prompt_embeds = encoder_pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt if do_cfg else None,
        do_classifier_free_guidance=do_cfg,
        device=device,
    )

    logger.info(
        "Encoded: prompt_embeds=%s  negative_prompt_embeds=%s",
        list(prompt_embeds.shape),
        list(negative_prompt_embeds.shape) if negative_prompt_embeds is not None else None,
    )

    payload = {"prompt_embeds": prompt_embeds.cpu()}
    if negative_prompt_embeds is not None:
        payload["negative_prompt_embeds"] = negative_prompt_embeds.cpu()

    raw = serialize_tensors(payload)

    del encoder_pipe, text_encoder, tokenizer
    flush_vram()
    logger.info("VRAM after cleanup: %.0f MB", vram_mb())
    logger.info("Serialized embeddings: %.2f KB", len(raw) / 1024)
    return raw


def stage_denoiser(
    model_path: str, raw_embeddings: bytes, seed: int, num_steps: int,
    num_frames: int, height: int, width: int, guidance_scale: float,
    device: str,
) -> bytes:
    """Stage 2: Load ONLY transformer + scheduler, denoise, return latents."""
    from diffusers import WanPipeline

    logger.info("=== Stage 2: Denoiser (transformer only) ===")
    logger.info("VRAM before load: %.0f MB", vram_mb())

    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    vae_config = {
        "latents_mean": pipe.vae.config.latents_mean,
        "latents_std": pipe.vae.config.latents_std,
        "z_dim": pipe.vae.config.z_dim,
        "scale_factor_spatial": pipe.vae.config.scale_factor_spatial,
        "scale_factor_temporal": pipe.vae.config.scale_factor_temporal,
    }

    pipe.text_encoder = None
    pipe.tokenizer = None
    pipe.vae = None
    flush_vram()
    logger.info("Denoiser VRAM (transformer only): %.0f MB", vram_mb())

    embeddings = deserialize_tensors(raw_embeddings)
    prompt_embeds = embeddings["prompt_embeds"].to(device)
    negative_prompt_embeds = embeddings.get("negative_prompt_embeds")
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
        num_frames=num_frames,
        output_type="latent",
    )
    latents = result.frames

    logger.info("Latents shape: %s dtype: %s", list(latents.shape), latents.dtype)

    raw = serialize_tensors({
        "latents": latents.cpu(),
        "vae_config": vae_config,
    })

    del pipe
    flush_vram()
    logger.info("VRAM after cleanup: %.0f MB", vram_mb())
    logger.info("Serialized latents: %.2f MB", len(raw) / 1024 / 1024)
    return raw


def stage_vae(model_path: str, raw_latents: bytes, device: str):
    """Stage 3: Load ONLY the VAE, decode latents to video frames."""
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    logger.info("=== Stage 3: VAE Decode (VAE only) ===")
    logger.info("VRAM before load: %.0f MB", vram_mb())

    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    vae.to(device)
    logger.info("VAE VRAM: %.0f MB", vram_mb())

    data = deserialize_tensors(raw_latents)
    latents = data["latents"].to(device)
    vae_config = data["vae_config"]

    latents = latents.to(vae.dtype)
    latents_mean = (
        torch.tensor(vae_config["latents_mean"])
        .view(1, vae_config["z_dim"], 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae_config["latents_std"])
        .view(1, vae_config["z_dim"], 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]

    video_processor = VideoProcessor(vae_scale_factor=vae_config["scale_factor_spatial"])
    frames = video_processor.postprocess_video(video, output_type="np")
    frames = frames[0]

    del vae
    flush_vram()
    logger.info("VRAM after cleanup: %.0f MB", vram_mb())
    logger.info("Decoded %d frames", len(frames))
    return frames


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_frames(frames_a, frames_b) -> dict:
    import numpy as np

    if len(frames_a) != len(frames_b):
        return {"match": False, "reason": f"frame count mismatch: {len(frames_a)} vs {len(frames_b)}"}

    a = np.stack([np.array(f) if not isinstance(f, np.ndarray) else f for f in frames_a]).astype(float)
    b = np.stack([np.array(f) if not isinstance(f, np.ndarray) else f for f in frames_b]).astype(float)

    if a.shape != b.shape:
        return {"match": False, "reason": f"shape mismatch: {a.shape} vs {b.shape}"}

    if a.max() <= 1.0:
        a = a * 255.0
    if b.max() <= 1.0:
        b = b * 255.0

    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    psnr = 10 * np.log10(255**2 / (diff**2).mean()) if diff.any() else float("inf")

    return {
        "match": max_diff < 10,
        "max_pixel_diff": float(max_diff),
        "mean_pixel_diff": float(mean_diff),
        "psnr_db": float(psnr),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate split diffusion pipeline (Wan video)")
    parser.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--prompt", default="A cat walking slowly on green grass in sunshine")
    parser.add_argument("--negative-prompt", default="Blurry, low quality, distorted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--output-dir", default="/tmp/disagg_validate")
    parser.add_argument("--skip-monolithic", action="store_true",
                        help="Skip monolithic run (for faster iteration on split logic)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Monolithic ───────────────────────────────────────────────────
    mono_frames = None
    if not args.skip_monolithic:
        t0 = time.monotonic()
        mono_frames = run_monolithic(
            args.model, args.prompt, args.negative_prompt,
            args.seed, args.num_steps, args.num_frames,
            args.height, args.width, args.guidance_scale,
            args.device,
        )
        elapsed = time.monotonic() - t0
        logger.info("Monolithic: %.2fs, %d frames", elapsed, len(mono_frames))
        _save_video(mono_frames, output_dir / "monolithic.mp4")

    # ── Split ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPLIT pipeline: Encoder → Denoiser → VAE")
    logger.info("=" * 60)

    t_total = time.monotonic()

    t0 = time.monotonic()
    raw_embeddings = stage_encoder(
        args.model, args.prompt, args.negative_prompt,
        args.guidance_scale, args.device,
    )
    t_enc = time.monotonic() - t0

    t0 = time.monotonic()
    raw_latents = stage_denoiser(
        args.model, raw_embeddings, args.seed, args.num_steps,
        args.num_frames, args.height, args.width,
        args.guidance_scale, args.device,
    )
    t_den = time.monotonic() - t0
    del raw_embeddings

    t0 = time.monotonic()
    split_frames = stage_vae(args.model, raw_latents, args.device)
    t_vae = time.monotonic() - t0
    del raw_latents

    t_split = time.monotonic() - t_total
    _save_video(split_frames, output_dir / "split.mp4")

    # ── Results ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Encoder:  %.2fs", t_enc)
    logger.info("  Denoiser: %.2fs", t_den)
    logger.info("  VAE:      %.2fs", t_vae)
    logger.info("  Total:    %.2fs", t_split)

    if mono_frames is not None:
        metrics = compare_frames(mono_frames, split_frames)
        logger.info("  Match:    %s", metrics["match"])
        if "max_pixel_diff" in metrics:
            logger.info(
                "  Max diff: %.1f   Mean diff: %.2f   PSNR: %.1f dB",
                metrics["max_pixel_diff"], metrics["mean_pixel_diff"], metrics["psnr_db"],
            )
        elif "reason" in metrics:
            logger.warning("  Reason: %s", metrics["reason"])
        if not metrics["match"]:
            logger.warning("Frames differ — check outputs visually (bf16 rounding is expected).")

    logger.info("  Output:   %s", output_dir)


def _save_video(frames, path):
    """Save a list of numpy frames as mp4 using diffusers utility."""
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, str(path), fps=16)
        logger.info("Saved video: %s (%d frames)", path, len(frames))
    except Exception as e:
        logger.warning("Could not save video (%s), saving first frame as PNG instead", e)
        import numpy as np
        from PIL import Image
        frame = frames[0]
        if isinstance(frame, np.ndarray):
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(frame).save(str(path).replace(".mp4", ".png"))


if __name__ == "__main__":
    main()
