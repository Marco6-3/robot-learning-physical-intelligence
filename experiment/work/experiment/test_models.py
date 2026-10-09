"""Numerical equivalence/causality tests plus an optional local GPU benchmark."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform

import torch
from einops import rearrange, repeat
from torch.nn import functional as F

from models import (PortableMamba1, associative_scan, sequential_scan, make_model,
                    count_parameters, optimizer_parameter_groups)


def upstream_scan_reference():
    # Parse only the pure Python reference function. Do not import or execute
    # upstream CUDA-extension imports, unrelated kernels, or network content.
    filename = Path(__file__).resolve().parents[1] / "references" / "selective_scan_interface.py"
    tree = ast.parse(filename.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "selective_scan_ref")
    namespace = {"torch": torch, "F": F, "rearrange": rearrange, "repeat": repeat}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), "exec"), namespace)
    return namespace["selective_scan_ref"]


def compare_scan_gradients(length: int, device: str):
    torch.manual_seed(831 + length)
    a = (torch.rand(2, length, 3, 4, device=device, dtype=torch.float64) * 0.2 + 0.79).requires_grad_()
    b = torch.randn_like(a).requires_grad_()
    weight = torch.randn_like(b)
    parallel, serial = associative_scan(a, b), sequential_scan(a, b)
    gp = torch.autograd.grad((parallel * weight).sum(), (a, b), retain_graph=True)
    gs = torch.autograd.grad((serial * weight).sum(), (a, b))
    torch.testing.assert_close(parallel, serial, rtol=1e-11, atol=1e-11)
    for p, s in zip(gp, gs):
        torch.testing.assert_close(p, s, rtol=1e-10, atol=1e-10)
    return {"length": length, "output_max_abs": (parallel - serial).abs().max().item(),
            "gradient_max_abs": max((p - s).abs().max().item() for p, s in zip(gp, gs))}


def compare_official_reference(length: int, device: str):
    torch.manual_seed(238 + length)
    mixer = PortableMamba1(8, d_state=8).to(device)
    x = torch.randn(2, length, 8, device=device, requires_grad=True)
    xz = mixer.in_proj(x)
    u, z = xz.chunk(2, dim=-1)
    u = F.silu(mixer.conv1d(u.transpose(1, 2))[..., :length]).transpose(1, 2)
    delta, b, c = mixer.x_proj(u).split([mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1)
    delta = F.linear(delta, mixer.dt_proj.weight)
    official = upstream_scan_reference()(
        u.transpose(1, 2), delta.transpose(1, 2), -mixer.A_log.exp(),
        b.transpose(1, 2), c.transpose(1, 2), mixer.D, z.transpose(1, 2),
        delta_bias=mixer.dt_proj.bias, delta_softplus=True)
    official = mixer.out_proj(official.transpose(1, 2))
    ours = mixer(x)
    weight = torch.randn_like(ours)
    parameters = (x, *tuple(mixer.parameters()))
    our_grad = torch.autograd.grad((ours * weight).sum(), parameters)
    ref_grad = torch.autograd.grad((official * weight).sum(), parameters)
    torch.testing.assert_close(ours, official, rtol=2e-4, atol=2e-6)
    for p, s in zip(our_grad, ref_grad):
        torch.testing.assert_close(p, s, rtol=3e-4, atol=1e-5)
    return {"length": length, "output_max_abs": (ours - official).abs().max().item(),
            "gradient_max_abs": max((p - s).abs().max().item() for p, s in zip(our_grad, ref_grad))}


def test_model_shapes_and_causality(device: str):
    result = {}
    for name in ("mlp", "tcn", "gru", "mamba"):
        torch.manual_seed(912)
        model = make_model(name, 8).to(device).eval()
        x = torch.randn(2, 64, 8, device=device)
        perturbed = x.clone()
        perturbed[:, 31:] = torch.randn_like(perturbed[:, 31:]) * 5
        with torch.no_grad():
            y, y_perturbed = model(x), model(perturbed)
        assert y.shape == (2, 64) and torch.isfinite(y).all()
        torch.testing.assert_close(y[:, :31], y_perturbed[:, :31], rtol=2e-5, atol=2e-6)
        model.train()
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        groups = optimizer_parameter_groups(model)
        assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))
        result[name] = {"parameters_input_dim8": count_parameters(model),
                        "future_perturbation_max_abs": (y[:, :31] - y_perturbed[:, :31]).abs().max().item()}
    mixer = PortableMamba1(32)
    dt = F.softplus(mixer.dt_proj.bias)
    assert dt.min() >= 0.001 and dt.max() <= 0.1
    torch.testing.assert_close(mixer.A_log.exp()[0], torch.arange(1, 9, dtype=torch.float32))
    return result


def benchmark(device: str, iterations: int = 20):
    if device != "cuda":
        return {"skipped": "GPU benchmark requested but CUDA unavailable"}
    output = {}
    for name in ("mlp", "tcn", "gru", "mamba"):
        model = make_model(name, 8).to(device)
        x = torch.randn(16, 128, 8, device=device)
        target = torch.rand(16, 128, device=device)
        model.train()
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(x), target).backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            model.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(x), target).backward()
        end.record()
        torch.cuda.synchronize()
        train_ms = start.elapsed_time(end) / iterations
        train_peak_bytes = torch.cuda.max_memory_allocated()
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(x)
            start.record()
            for _ in range(iterations):
                model(x)
            end.record()
            torch.cuda.synchronize()
        output[name] = {"parameters": count_parameters(model), "forward_ms": start.elapsed_time(end) / iterations,
                        "forward_backward_ms": train_ms, "peak_allocated_MiB": train_peak_bytes / 2**20,
                        "incremental_peak_MiB": (train_peak_bytes - baseline_bytes) / 2**20}
        del model, x, target
        torch.cuda.empty_cache()
    return {"batch": 16, "length": 128, "input_dim": 8, "dtype": "float32", "iterations": iterations,
            "note": "Portable non-fused Mamba latency/memory are not fair architecture speed comparisons.",
            "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Deterministic fp32 precision for direct independent-equation comparisons.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"torch": torch.__version__, "platform": platform.platform(), "device": args.device,
              "device_name": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
              "scan_oracle": [compare_scan_gradients(length, args.device) for length in (17, 64)],
              "official_reference": [compare_official_reference(length, args.device) for length in (17, 64)],
              "models": test_model_shapes_and_causality(args.device)}
    if args.benchmark:
        report["benchmark"] = benchmark(args.device)
    report["status"] = "all checks passed"
    destination = Path(__file__).with_name("model_validation.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
