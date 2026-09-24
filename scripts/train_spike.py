#!/usr/bin/env python
"""Measure training speed for a preset."""

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

# so forge imports work from here
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import torch  # noqa: E402

from forge.train.model import GPT  # noqa: E402
from forge.train.presets import PRESETS, count_params, get_preset  # noqa: E402

FALLBACK_URL = "https://www.gutenberg.org/files/11/11-0.txt"


def load_text(path: Path | None, cache_dir: Path) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8", errors="replace")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "alice.txt"
    if not cached.exists():
        print(f"no --data given, downloading a public domain book from {FALLBACK_URL}")
        try:
            with urllib.request.urlopen(FALLBACK_URL, timeout=60) as response:
                cached.write_bytes(response.read())
        except Exception as problem:
            raise SystemExit(
                f"couldn't download the sample corpus ({problem}).\n"
                "pass your own text file instead:  --data path/to/corpus.txt"
            ) from problem
    return cached.read_text(encoding="utf-8", errors="replace")


def make_byte_tokens(text: str, vocab_size: int) -> torch.Tensor:
    # only vocab size changes the speed
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 for the byte stand-in")
    # torch wants a writable buffer
    return torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).long()


def get_batch(
    tokens: torch.Tensor, batch_size: int, ctx_len: int, device: str, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(tokens) - ctx_len - 1, (batch_size,), generator=generator, device="cpu"
    )
    x = torch.stack([tokens[i : i + ctx_len] for i in starts])
    y = torch.stack([tokens[i + 1 : i + 1 + ctx_len] for i in starts])
    if device == "cuda":
        # overlap the copy with compute
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype | None:
    if device != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_one(args, preset_name: str, text: str) -> dict:
    device = pick_device(args.device)
    autocast_dtype = pick_dtype(device)
    config = get_preset(preset_name, ctx_len=args.ctx_len, vocab_size=args.vocab_size)

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    tokens = make_byte_tokens(text, config.vocab_size)
    if len(tokens) < config.ctx_len + 2:
        raise SystemExit(f"corpus is too small: {len(tokens)} tokens, need > {config.ctx_len + 2}")

    model = GPT(config).to(device)
    optimizer = model.configure_optimizer(lr=args.lr)

    measured_params = model.num_params()
    formula_params = count_params(config)
    assert measured_params == formula_params, (
        f"param count mismatch: model={measured_params} formula={formula_params}"
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(
        f"\n=== {preset_name}: {config.n_layer}L x {config.d_model}d x {config.n_head}h, "
        f"ctx {config.ctx_len}, vocab {config.vocab_size}"
    )
    print(f"    params: {measured_params:,} ({measured_params / 1e6:.2f}M)")
    print(f"    device: {device}, autocast: {autocast_dtype or 'off (fp32)'}")
    print(f"    batch: {args.batch_size} seqs x {config.ctx_len} tokens")
    print(f"    loss at init should be about ln({config.vocab_size}) = "
          f"{math.log(config.vocab_size):.2f}")

    scaler = torch.amp.GradScaler(device, enabled=autocast_dtype == torch.float16)
    step_times: list[float] = []
    losses: list[float] = []
    history: list[dict] = []
    model.train()

    for step in range(1, args.steps + 1):
        x, y = get_batch(tokens, args.batch_size, config.ctx_len, device, generator)

        step_start = time.perf_counter()
        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise SystemExit(f"loss went to {loss_value} at step {step}. That's a bug, not a speed")

        losses.append(loss_value)
        history.append({"step": step, "loss": round(loss_value, 4)})
        # cuda autotuning makes early steps lie
        if step > args.warmup_steps:
            step_times.append(elapsed)

        if step % args.log_every == 0 or step == 1:
            print(f"    step {step:>4}/{args.steps}  loss {loss_value:.4f}  {elapsed * 1000:.0f} ms")

    if not step_times:
        raise SystemExit("every step was warmup. Use --steps larger than --warmup-steps")

    tokens_per_step = args.batch_size * config.ctx_len
    median_step = statistics.median(step_times)
    tokens_per_sec = tokens_per_step / median_step

    peak_vram = None
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated()

    first_loss = losses[0]
    last_loss = statistics.mean(losses[-min(10, len(losses)) :])
    result = {
        "preset": preset_name,
        "config": asdict(config),
        "params": measured_params,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "autocast": str(autocast_dtype) if autocast_dtype else "fp32",
        "batch_size": args.batch_size,
        "steps": args.steps,
        "first_loss": round(first_loss, 4),
        "final_loss_mean_last10": round(last_loss, 4),
        "loss_went_down": last_loss < first_loss,
        "median_step_seconds": round(median_step, 5),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_bytes": peak_vram,
        "peak_vram_gb": round(peak_vram / 1e9, 3) if peak_vram else None,
        "loss_history": history,
    }

    print(f"    -> loss {first_loss:.3f} -> {last_loss:.3f}")
    print(f"    -> {tokens_per_sec:,.0f} tokens/sec (median step {median_step * 1000:.0f} ms)")
    if peak_vram:
        print(f"    -> peak VRAM {peak_vram / 1e9:.2f} GB")
    for budget in (100e6, 300e6, 1e9):
        hours = budget / tokens_per_sec / 3600
        print(f"    -> {budget / 1e6:,.0f}M tokens would take about {hours:.1f} h at this rate")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="nano",
        help=f"one of {sorted(PRESETS)}, or 'all' to sweep every preset",
    )
    parser.add_argument("--data", type=Path, help="a plain text file to train on")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=3, help="steps excluded from timing")
    parser.add_argument("--batch-size", type=int, default=8, help="sequences per step")
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=Path, help="write the results here as json")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run to write results (they are NOT a GPU budget)",
    )
    args = parser.parse_args()

    cache_dir = Path(__file__).resolve().parent / ".spike-cache"
    text = load_text(args.data, cache_dir)
    print(f"corpus: {len(text):,} characters")

    presets = sorted(PRESETS) if args.preset == "all" else [args.preset]
    results = [run_one(args, name, text) for name in presets]

    print("\n=== summary ===")
    device = results[0]["device"]
    gpu = results[0]["gpu_name"] or "CPU"
    print(f"hardware: {gpu}")
    header = f"{'preset':<8} {'params':>12} {'tok/sec':>12} {'peak VRAM':>11} {'300M tokens':>13}"
    print(header)
    for r in results:
        vram = f"{r['peak_vram_gb']:.2f} GB" if r["peak_vram_gb"] else "n/a"
        hours = 300e6 / r["tokens_per_sec"] / 3600
        print(
            f"{r['preset']:<8} {r['params']:>12,} {r['tokens_per_sec']:>12,.0f} "
            f"{vram:>11} {hours:>11.1f} h"
        )
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit(
            "\nthis ran on CPU, so these are NOT our GPU numbers and nothing was written.\n"
            "on the windows GPU machine, install the CUDA build first:\n"
            "  uv sync --extra train\n"
            "  uv pip install --reinstall torch --index-url "
            "https://download.pytorch.org/whl/cu130\n"
            "  uv run python -c \"import torch; print(torch.cuda.is_available())\"\n"
            "that last line must print True. then re-run this command.\n"
            "to record a CPU run anyway (plumbing checks only), pass --allow-cpu."
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    if device != "cuda":
        print("\nNOTE: CPU run (--allow-cpu). These numbers are NOT a GPU budget.")


if __name__ == "__main__":
    main()
