"""Benchmark the frozen final models only when the GPU is explicitly available.

This script does not train on experiment data or alter any checkpoints. It
measures fresh random models on random tensors, in a separate process, and saves
timing/configuration metadata. Portable Mamba is unfused: no result from this
script establishes architecture speed or memory efficiency.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from models import make_model, count_parameters


def run_case(name: str, batch: int, length: int, input_dim: int,
             warmup: int, repeats: int, iterations: int) -> dict:
    torch.manual_seed(481)
    torch.cuda.empty_cache()
    model = make_model(name, input_dim).cuda()
    x = torch.randn(batch, length, input_dim, device="cuda")
    target = torch.randint(0, 2, (batch, length), device="cuda", dtype=torch.int64).float()
    result = {"model": name, "config": model.config, "parameters": count_parameters(model),
              "batch": batch, "length": length, "input_dim": input_dim}
    for mode in ("inference_forward", "training_forward_backward"):
        training = mode == "training_forward_backward"
        model.train(training)

        def operation():
            if training:
                model.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(model(x), target).backward()
            else:
                with torch.inference_mode():
                    model(x)

        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        base_bytes = torch.cuda.memory_allocated()
        event_means, wall_means = [], []
        import time
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(iterations):
                operation()
            end.record()
            torch.cuda.synchronize()
            event_means.append(start.elapsed_time(end) / iterations)
            wall_means.append((time.perf_counter() - wall_start) * 1000 / iterations)
        peak_bytes = torch.cuda.max_memory_allocated()
        result[mode] = {
            "cuda_ms_per_batch_median": statistics.median(event_means),
            "cuda_ms_per_batch_repeats": event_means,
            "synchronized_wall_ms_per_batch_median": statistics.median(wall_means),
            "synchronized_wall_ms_per_batch_repeats": wall_means,
            "cuda_ms_per_frame_amortized": statistics.median(event_means) / (batch * length),
            "base_allocated_bytes": base_bytes,
            "peak_allocated_bytes": peak_bytes,
            "incremental_peak_bytes": peak_bytes - base_bytes,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="work/results/final_model_benchmark.json")
    parser.add_argument("--input-dim", type=int, default=54)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an available CUDA GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "python_platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(), "dtype": "float32", "tf32": False,
        "threads": torch.get_num_threads(), "warmup": args.warmup,
        "repeats": args.repeats, "iterations_per_repeat": args.iterations,
        "optimizer_step_included": False, "results": [],
        "interpretation": [
            "Implementation-specific timing on random fresh models; not experimental prediction performance.",
            "Portable non-fused Mamba has O(L log L) scan work and cannot establish architecture speed superiority.",
            "B1 still evaluates a full L128 window; it is not cached single-step streaming inference latency.",
            "Per-frame amortized timings are throughput conversions, not event detection delay.",
            "Peak memory includes tensors and model allocations in this isolated benchmark process.",
        ],
    }
    for batch in (1, 16):
        for name in ("mlp", "tcn", "gru", "mamba"):
            result = run_case(name, batch, args.length, args.input_dim,
                              args.warmup, args.repeats, args.iterations)
            report["results"].append(result)
            print(json.dumps({"finished": name, "batch": batch,
                              "forward_ms": result["inference_forward"]["cuda_ms_per_batch_median"]}), flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
