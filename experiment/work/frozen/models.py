# Mamba mixer formulation/initialization adapted from state-spaces/mamba.
# Copyright (c) 2023, Tri Dao, Albert Gu. Licensed under Apache-2.0.
# See ../references/mamba_LICENSE and mamba_source_manifest.json.
"""Causal, sequence-to-sequence binary classifiers for controlled experiments.

All public models consume [batch, time, input_dim] and emit [batch, time] logits.
The portable Mamba-1 mixer follows upstream state-spaces/mamba at commit
e9594ce1c732d97440f0332fdc43170a2294dbfa (Apache-2.0; references/ contains the
original source, license and hashes). The upstream mixer formulation and special
parameter initialization are retained; an unfused PyTorch associative scan
replaces its CUDA scan. Residual pre-LayerNorm blocks wrap the mixer.

This is NOT the official fused CUDA implementation: its measured latency and
memory are implementation-specific and cannot establish architecture efficiency.
Input-dependent delta, B and C are computed at every step. States start at zero
for every forward call; split sequences therefore do not carry state implicitly.
"""
from __future__ import annotations

import math
import inspect
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def associative_scan(a: Tensor, b: Tensor) -> Tensor:
    """Inclusive h[t] = a[t]*h[t-1] + b[t], h[-1]=0; time is axis 1.

    Hillis-Steele doubling composes affine maps (a,b) without division. This is
    differentiable, O(log L) launch depth and O(L log L) arithmetic, unlike the
    work-efficient fused upstream scan. No in-place writes touch autograd data.
    """
    if a.shape != b.shape or a.ndim < 2 or a.shape[1] == 0:
        raise ValueError("a,b must have identical shapes and a nonempty time axis 1")
    offset = 1
    length = a.shape[1]
    while offset < length:
        b_next = torch.cat((b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]), dim=1)
        if offset * 2 < length:
            a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), dim=1)
        b = b_next
        offset *= 2
    return b


def sequential_scan(a: Tensor, b: Tensor) -> Tensor:
    """Independent chronological oracle, intentionally not used for training."""
    state = torch.zeros_like(b[:, 0])
    states = []
    for time in range(b.shape[1]):
        state = a[:, time] * state + b[:, time]
        states.append(state)
    return torch.stack(states, dim=1)


class PortableMamba1(nn.Module):
    """Real-valued Mamba-1 mixer, with its reference discretization and gate."""

    def __init__(self, d_model: int, d_state: int = 8, expand: int = 2,
                 d_conv: int = 4, dt_rank: int | str = "auto",
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32)
                                  .log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, inputs: Tensor, scan: Callable[[Tensor, Tensor], Tensor] = associative_scan) -> Tensor:
        x, z = self.in_proj(inputs).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :inputs.shape[1]].transpose(1, 2))
        dt_raw, b_t, c_t = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Match official fp32 state arithmetic under AMP, retaining fp64 for QA.
        scan_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        delta = F.softplus(self.dt_proj(dt_raw).to(scan_dtype))
        x_scan, b_t, c_t = x.to(scan_dtype), b_t.to(scan_dtype), c_t.to(scan_dtype)
        a = -torch.exp(self.A_log.to(scan_dtype))
        delta_a = torch.exp(delta.unsqueeze(-1) * a)
        delta_b_u = delta.unsqueeze(-1) * b_t.unsqueeze(-2) * x_scan.unsqueeze(-1)
        states = scan(delta_a, delta_b_u)
        y = (states * c_t.unsqueeze(-2)).sum(dim=-1) + self.D.to(scan_dtype) * x_scan
        y = y * F.silu(z.to(scan_dtype))
        return self.out_proj(y.to(x.dtype))


class MambaBlock(nn.Module):
    def __init__(self, width: int, d_state: int = 8, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mixer = PortableMamba1(width, d_state=d_state, expand=expand, d_conv=d_conv)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


class MambaClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int = 34, depth: int = 2,
                 d_state: int = 8, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(MambaBlock(width, d_state, expand, d_conv) for _ in range(depth)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.norm(self.blocks(self.input_proj(x)))).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int = 64, embed_dim: int = 32, depth: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, width, num_layers=depth, batch_first=True)
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        states, _ = self.gru(self.input_proj(x))
        return self.head(states).squeeze(-1)


class CausalTCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(self.norm(x).transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        return x + F.silu(y)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int = 39, depth: int = 4, kernel_size: int = 3):
        super().__init__()
        self.receptive_field = 1 + (kernel_size - 1) * sum(2 ** i for i in range(depth))
        self.input_proj = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(CausalTCNBlock(width, 2 ** i, kernel_size) for i in range(depth)))
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.blocks(self.input_proj(x))).squeeze(-1)


class MLPClassifier(nn.Module):
    """Pointwise baseline: every output sees only the current input frame."""
    def __init__(self, input_dim: int, width: int = 122, depth: int = 2, embed_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        layers: list[nn.Module] = [nn.Linear(embed_dim, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.input_proj(x)).squeeze(-1)


def make_model(name: str, input_dim: int, width: int | None = None, **kwargs) -> nn.Module:
    constructors = {"mlp": MLPClassifier, "tcn": TCNClassifier,
                    "gru": GRUClassifier, "mamba": MambaClassifier}
    try:
        constructor = constructors[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; expected one of {list(constructors)}") from exc
    if width is not None:
        kwargs["width"] = width
    model = constructor(input_dim=input_dim, **kwargs)
    configuration = inspect.signature(constructor).bind(input_dim=input_dim, **kwargs)
    configuration.apply_defaults()
    model.config = {"name": name.lower(), **configuration.arguments}
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def optimizer_parameter_groups(model: nn.Module, weight_decay: float = 0.01) -> list[dict]:
    """Honor upstream A/D no-decay flags; biases/norm parameters also get no decay."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
