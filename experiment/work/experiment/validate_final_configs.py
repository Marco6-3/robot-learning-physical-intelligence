"""Fast CPU validation after parameter-budget matching; no GPU work."""
import json
from pathlib import Path
import torch
from models import make_model, count_parameters

torch.set_num_threads(4)
report = {"device": "cpu", "inputs": {}, "note": "Final matched defaults; prior GPU timing used earlier widths and is not final-model timing."}
for input_dim in (8, 54):
    report["inputs"][str(input_dim)] = {}
    for name in ("mlp", "tcn", "gru", "mamba"):
        torch.manual_seed(912)
        model = make_model(name, input_dim).eval()
        x = torch.randn(2, 64, input_dim)
        perturbed = x.clone()
        perturbed[:, 31:] = torch.randn_like(perturbed[:, 31:]) * 5
        with torch.no_grad():
            y, yp = model(x), model(perturbed)
        assert y.shape == (2, 64)
        torch.testing.assert_close(y[:, :31], yp[:, :31], rtol=2e-5, atol=2e-6)
        model.train()
        model(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        report["inputs"][str(input_dim)][name] = {"config": model.config, "parameters": count_parameters(model),
                                                  "future_perturbation_max_abs": float((y[:, :31] - yp[:, :31]).abs().max())}
    counts = [item["parameters"] for item in report["inputs"][str(input_dim)].values()]
    assert max(counts) / min(counts) < 1.05
report["status"] = "all final-config shape, causality, finite-gradient, parameter-budget checks passed"
Path(__file__).with_name("model_final_configs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
