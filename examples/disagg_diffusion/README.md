# Disaggregated Diffusion Inference POC

Split a monolithic video diffusion pipeline (Text Encoder → Transformer → VAE) into
independent stages on separate GPUs. Tensor data transfers between stages use
**NIXL RDMA** (GPU-direct); only small metadata travels over Dynamo RPC.

Design doc: [docs/design/disaggregated_diffusion.md](../../docs/design/disaggregated_diffusion.md)

## Architecture

```
GPU 0: Encoder Worker  ──NIXL RDMA──→  GPU 1: Denoiser Worker  ──NIXL RDMA──→  GPU 2: VAE Worker
                                ↕ metadata (RPC)                        ↕ metadata (RPC)
                                    Orchestrator (no GPU)
```

## Quick Start

```bash
conda activate omni
export HF_HUB_CACHE=/path/to/huggingface/hub

# All-in-one (starts 3 workers + orchestrator)
bash launch/run_all.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers "A cat walking on grass"
```

### Manual Launch

```bash
# Terminal 0: etcd (service discovery)
etcd --data-dir /tmp/etcd_disagg --listen-client-urls http://0.0.0.0:2379

# Terminal 1: Encoder Worker (~11 GB VRAM)
CUDA_VISIBLE_DEVICES=0 python phase1_workers/encoder_worker.py

# Terminal 2: Denoiser Worker (~10 GB VRAM)
CUDA_VISIBLE_DEVICES=1 python phase1_workers/denoiser_worker.py

# Terminal 3: VAE Worker (~3 GB VRAM)
CUDA_VISIBLE_DEVICES=2 python phase1_workers/vae_worker.py

# Terminal 4: Orchestrator (no GPU)
PROMPT="A golden retriever on a beach" NUM_STEPS=20 OUTPUT=/tmp/video.mp4 \
    python phase2_orchestrator/run_disagg.py
```

## Phases

### Phase 0: Offline Validation (no Dynamo)

Single-GPU script proving diffusers supports split execution.

```bash
python phase0_validate/validate_split.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A cat walking on grass" \
    --num-steps 10 --num-frames 17 \
    --output-dir /tmp/disagg_validate
```

### Phase 1: Dynamo Stage Workers (NIXL)

Three Dynamo workers with NIXL RDMA tensor transfer:

| Worker | Model Component | VRAM | Endpoint |
|--------|----------------|------|----------|
| `encoder_worker.py` | UMT5 text encoder | ~11 GB | `disagg_diffusion.encoder.generate` |
| `denoiser_worker.py` | WanTransformer3DModel | ~10 GB | `disagg_diffusion.denoiser.generate` |
| `vae_worker.py` | AutoencoderKLWan (fp32) | ~3 GB | `disagg_diffusion.vae.generate` |

### Phase 2: Orchestrator (Pipeline Parallel)

Chains three stage endpoints with pipeline parallelism. Multiple concurrent
requests overlap across stages — while request 1 is in the Denoiser, request 2
can run in the Encoder simultaneously:

```
Request 1:  [Encoder] → [Denoiser] → [  VAE  ]
Request 2:             [Encoder] → [Denoiser] → [  VAE  ]
Request 3:                        [Encoder] → [Denoiser] → ...
```

Key features:
- **Per-stage semaphores** — one GPU per stage, backpressure via `asyncio.Semaphore(1)`
- **Admission control** — `MAX_PIPELINE_DEPTH` (default 4) limits total in-flight requests
- **No base64 over RPC** — VAE writes mp4 to shared storage, only the filename goes over RPC
- **Pipeline status** — `GET /pipeline/status` shows per-stage utilization

Environment variables:
- `MAX_PIPELINE_DEPTH` — max concurrent requests in pipeline (default: 4)
- `OUTPUT_DIR` — shared directory for video output (default: `/tmp/disagg_videos`)

## Results

Wan2.2-TI2V-5B, 480×832, 17 frames, 3 GPUs:

| Steps | Encoder | Denoiser | VAE | Total |
|-------|---------|----------|-----|-------|
| 10 | 2.5s | 8.4s | 5.7s | **16.6s** |
| 20 | 0.1s | 8.9s | 3.5s | **12.5s** |
| 50 | 0.1s | 43.8s | 3.2s | **47.1s** |

## Dependencies

```bash
pip install ai-dynamo-runtime diffusers transformers ftfy imageio imageio-ffmpeg etcd-distro
```

## Supported Models

Any `diffusers.WanPipeline` model with `encode_prompt()` + `output_type="latent"`:

- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (recommended, 5B)
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`
