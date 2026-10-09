"""Publication-style Chinese figures from complete, checked analysis JSON only.

No models are run and no statistics are invented/recomputed here. Five-seed
standard deviations are labeled as SD, never confidence intervals. All three
analysis files must exist and contain the complete registered model grids.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COLORS = {"mlp": "#737B84", "tcn": "#C68B2B", "gru": "#2179A6", "mamba": "#C65058"}
LABELS = {"mlp": "MLP", "tcn": "TCN", "gru": "GRU", "mamba": "Mamba-1"}
MARKERS = {"mlp": "o", "tcn": "s", "gru": "D", "mamba": "^"}
OBJECT_LABELS = {"SpoolSolder": "焊锡卷", "brush": "刷子", "screwDriver": "螺丝刀"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_style():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.is_file():
        font_path = Path(font_manager.findfont("Microsoft YaHei", fallback_to_default=False))
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": font_name, "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7C8794", "axes.linewidth": .65,
        "grid.color": "#DDE3E9", "grid.linewidth": .55,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 1.45, "lines.markersize": 5.5,
    })
    return {"font_name": font_name, "font_path": str(font_path)}


def load_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report.get("cells"), list) or not report["cells"]:
        raise ValueError(f"No checked cell summaries found: {path}")
    for row in report["cells"]:
        if row.get("seeds") != 5:
            raise ValueError(f"Expected five seeds per cell: {path}, {row}")
        for metric in ("ap_mean", "ap_seed_sd"):
            if row.get(metric) is None or not np.isfinite(row[metric]):
                raise ValueError(f"Missing/nonfinite {metric}: {path}")
        if not 0 <= row["ap_mean"] <= 1 or row["ap_seed_sd"] < 0:
            raise ValueError(f"Invalid AP summary: {path}")
    return report


def validate_factorial(report, models, sizes):
    lookup = {}
    for row in report["cells"]:
        key = (row["rate"], row["lag"], row["n_train"], row["model"])
        if key in lookup:
            raise ValueError(f"Duplicate cell {key}")
        lookup[key] = row
    expected = {(rate, lag, n, model) for rate in (.01, .20) for lag in (2, 64)
                for n in sizes for model in models}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete or unexpected factorial grid. Missing={expected-set(lookup)}, extra={set(lookup)-expected}")
    for row in lookup.values():
        for field in ("query_ap_mean", "query_ap_seed_sd"):
            if row.get(field) is None or not np.isfinite(row[field]):
                raise ValueError(f"Missing required query diagnostic {field}")
    return lookup


def axis_format(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(length=3, width=.65)


def baseline_value(report, rate, lag, metric):
    candidates = [r for r in report.get("baselines", [])
                  if np.isclose(r.get("rate", -1), rate) and ("lag" not in r or r["lag"] == lag)]
    fields = (["score_query_all_frame_ap", "query_only_baseline_ap", "query_baseline_ap"]
              if metric == "ap" else ["query_only_constant_score_ap", "query_conditioned_positive_fraction", "query_baseline_ap"])
    for row in candidates:
        for field in fields:
            if field in row and row[field] is not None:
                return float(row[field])
    raise ValueError(f"Missing source-derived query-only baseline for rate={rate},lag={lag},metric={metric}; never substitute .5 silently")


def save_figure(fig, output, stem, title, artifacts):
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        kwargs = {"dpi": 300} if extension == "png" else {"metadata": {"Title": title, "Creator": "Matplotlib source-backed experiment figures"}}
        fig.savefig(path, bbox_inches="tight", pad_inches=.12, **kwargs)
        artifacts.append({"path": path.name, "sha256": digest(path)})
    plt.close(fig)


def factorial_plot(report, lookup, models, sizes, version, metric, output, artifacts):
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.4), sharex=True, sharey=True)
    means = np.array([r[metric + "_mean"] for r in lookup.values()])
    sds = np.array([r[metric + "_seed_sd"] for r in lookup.values()])
    lower, upper = min(-.025, float(np.min(means-sds))-.025), max(1.025, float(np.max(means+sds))+.025)
    offsets = np.linspace(-.105, .105, len(models))
    for ri, rate in enumerate((.01, .20)):
        for ci, lag in enumerate((2, 64)):
            ax = axes[ri, ci]
            axis_format(ax)
            for offset, model in zip(offsets, models):
                rows = [lookup[(rate, lag, n, model)] for n in sizes]
                ax.errorbar(np.arange(2) + offset, [r[metric + "_mean"] for r in rows],
                            yerr=[r[metric + "_seed_sd"] for r in rows], color=COLORS[model],
                            marker=MARKERS[model], capsize=3, elinewidth=1, label=LABELS[model], zorder=4)
            baseline = baseline_value(report, rate, lag, metric)
            ax.axhline(baseline, color="#9AA3AB", ls=(0, (4, 3)), lw=1, zorder=1)
            ax.text(.98, baseline + .022, f"查询基线 {baseline:.3f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color="#646F79")
            rarity = "稀疏" if rate == .01 else "密集"
            memory = "短跨度" if lag == 2 else "长跨度"
            ax.set_title(f"{rarity}（名义阳性率 {rate:.0%}） · {memory}（{lag} 步）", loc="left", pad=9)
            ax.set_xticks([0, 1], [f"{n:,}" for n in sizes])
            ax.set_xlim(-.24, 1.24)
            ax.set_ylim(lower, upper)
            if ri == 1:
                ax.set_xlabel("训练记录数")
            if ci == 0:
                ax.set_ylabel(("全评分时间步 AUPRC" if version == "v1" else "记录终点 AUPRC") if metric == "ap" else "查询条件 AUPRC")
    task = "v1：固定延迟随机符号检索" if version == "v1" else "v2：单个相关位的保持"
    meaning = ("仅 query=1 的评分位置" if metric == "query_ap" else
               "每条序列的全部有效时间步评分" if version == "v1" else "每记录仅终点评分；阳性率按记录计")
    title = task + (" · AUPRC" if metric == "ap" else " · 查询条件 AUPRC")
    fig.suptitle(title, x=.085, y=.995, ha="left", fontsize=15, fontweight="bold")
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.63, .965), ncol=len(models), frameon=False)
    fig.text(.085, .013, "点：5 个训练种子均值；误差棒：5 个种子的标准差（非置信区间）。\n"
             + meaning + "；虚线为实际测试样本计算的查询基线。仅两个数据预算，连线用于导视。", fontsize=8.5, color="#4D5863")
    fig.subplots_adjust(left=.085, right=.985, top=.865, bottom=.145, hspace=.35, wspace=.17)
    save_figure(fig, output, f"{version}_{metric}", title, artifacts)


def real_plots(report, output, artifacts):
    held = [r for r in report["cells"] if r["suite"] == "held_trial"]
    loo = [r for r in report["cells"] if r["suite"] == "loo"]
    if len(held) != 8 or len(loo) != 6:
        raise ValueError("Real plots require all 8 held-trial and 6 fixed-object LOO cells")
    held_sizes = sorted({r["n_train_trials"] for r in held})
    if held_sizes != [18, 54]:
        raise ValueError(f"Unexpected registered held-trial sizes: {held_sizes}")
    held_lookup = {(r["n_train_trials"], r["model"]): r for r in held}
    models = ["mlp", "tcn", "gru", "mamba"]
    if len(held_lookup) != 8 or any((n, m) not in held_lookup for n in held_sizes for m in models):
        raise ValueError("Duplicate/missing held-trial cell")
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.2]})
    for ax in axes:
        axis_format(ax)
        ax.set_ylim(min(-.02, min(r["ap_mean"]-r["ap_seed_sd"] for r in report["cells"])-.02),
                    max(1.02, max(r["ap_mean"]+r["ap_seed_sd"] for r in report["cells"])+.02))
    for offset, model in zip(np.linspace(-.10, .10, 4), models):
        rows = [held_lookup[(n, model)] for n in held_sizes]
        axes[0].errorbar(np.arange(2)+offset, [r["ap_mean"] for r in rows],
                         yerr=[r["ap_seed_sd"] for r in rows], color=COLORS[model], marker=MARKERS[model], capsize=3)
    axes[0].set_xticks([0, 1], ["18", "54"])
    axes[0].set_xlim(-.25, 1.25)
    axes[0].set_xlabel("训练试次数")
    axes[0].set_ylabel("真实触觉检测 AUPRC")
    axes[0].set_title("A  同物体、留出完整试次", loc="left", pad=10)
    configs = sorted({r["configuration"] for r in loo})
    if len(configs) != 3:
        raise ValueError("Expected exactly three fixed-object LOO configurations")
    loo_lookup = {(r["configuration"], r["model"]): r for r in loo}
    for offset, model in zip((-.045, .045), ("gru", "mamba")):
        rows = [loo_lookup[(c, model)] for c in configs]
        axes[1].errorbar(np.arange(3)+offset, [r["ap_mean"] for r in rows],
                         yerr=[r["ap_seed_sd"] for r in rows], color=COLORS[model], marker=MARKERS[model],
                         linestyle="none", capsize=3)
    axes[1].set_xticks(np.arange(3), [OBJECT_LABELS.get(c.removeprefix("loo_"), c.removeprefix("loo_")) for c in configs])
    axes[1].set_xlim(-.4, 2.4)
    axes[1].set_xlabel("留出的固定物体")
    axes[1].set_title("B  留一物体（3 个固定对象）", loc="left", pad=10)
    baseline = {r["configuration"]: r["ap"] for r in report.get("always_alarm_baselines", [])}
    held_configs = [held_lookup[(n, "gru")]["configuration"] for n in held_sizes]
    if any(c not in baseline for c in held_configs+configs):
        raise ValueError("Missing actual real-data prevalence baselines")
    axes[0].plot([0, 1], [baseline[c] for c in held_configs], ls=(0, (4, 3)), color="#929DA7", lw=1)
    axes[1].scatter(np.arange(3), [baseline[c] for c in configs], marker="_", s=500, color="#929DA7", lw=1.2)
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], label=LABELS[m]) for m in models]
    handles.append(Line2D([0], [0], color="#929DA7", ls="--", label="恒定分数基线（阳性率）"))
    title = "真实触觉数据：留出试次与留一物体"
    fig.suptitle(title, x=.075, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .93), ncol=5, frameon=False)
    fig.text(.075, .027, "点：5 个训练种子均值；误差棒：5 个种子的标准差（非置信区间）。\n"
             "此图为当前滑动检测；自然事件率，不能直接证明 1% 稀疏情境或长记忆机制。3 个留出物体不代表新物体总体。",
             fontsize=8.5, color="#4D5863")
    fig.subplots_adjust(left=.075, right=.985, top=.78, bottom=.22, wspace=.18)
    save_figure(fig, output, "real_ap", title, artifacts)


def contrast_plot(synthetic, cue, output, artifacts):
    reports = [("v1 固定延迟检索", synthetic), ("v2 单相关位保持", cue)]
    if not all(any(r.get("metric") == "ap" for r in report.get("hypotheses", [])) for _, report in reports):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (label, report) in zip(axes, reports):
        rows = [r for r in report["hypotheses"] if r.get("metric") == "ap"]
        if len(rows) != 3 or any(r.get(k) is None for r in rows for k in ("estimate", "lower", "upper", "confidence")):
            plt.close(fig)
            return False
        for i, row in enumerate(rows):
            ax.hlines(i, row["lower"], row["upper"], color="#3F708E", lw=2)
            ax.scatter(row["estimate"], i, marker="D", color="#21495F", s=34, zorder=4)
        ax.set_yticks(range(3), [f"{r['hypothesis']}\n区间水平 {r['confidence']:.2%}" for r in rows])
        ax.invert_yaxis()
        ax.set_ylim(2.5, -.5)
        ax.axvline(0, color="#939DA5", lw=1, ls="--")
        ax.grid(axis="x", zorder=0)
        ax.set_title(label, loc="left")
        ax.set_xlabel("已定义的 AUPRC 对比量")
    title = "假设对比：点估计与已计算的重采样区间"
    fig.suptitle(title, x=.08, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.08, .015, "使用分析文件中已计算的区间，不在绘图时重算。H1/H2/H3 的方向和对比定义不同，详见分析表。\n"
             "区间条件于既定划分、训练预算和超参数选择；v2 为另一机制任务，与 v1 复用基础随机流。", fontsize=8.5, color="#4D5863")
    fig.subplots_adjust(left=.10, right=.985, top=.80, bottom=.24, wspace=.43)
    save_figure(fig, output, "hypothesis_intervals", title, artifacts)
    return True


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, default=ROOT/"work/results/synthetic_analysis/analysis.json")
    parser.add_argument("--real", type=Path, default=ROOT/"work/results/real_analysis/analysis.json")
    parser.add_argument("--cue", type=Path, default=ROOT/"work/results/cue_memory_v2_analysis/analysis.json")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/figures")
    parser.add_argument("--include-contrasts", action="store_true")
    args = parser.parse_args()
    paths = {"v1": args.synthetic, "real": args.real, "v2": args.cue}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("All three final analyses must exist before figure generation: " + "; ".join(missing))
    reports = {key: load_report(path) for key, path in paths.items()}
    v1 = validate_factorial(reports["v1"], ["mlp", "tcn", "gru", "mamba"], (64, 256))
    v2 = validate_factorial(reports["v2"], ["gru", "mamba"], (4096, 16384))
    style = configure_style()
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for metric in ("ap", "query_ap"):
        factorial_plot(reports["v1"], v1, ["mlp", "tcn", "gru", "mamba"], (64, 256), "v1", metric, args.out, artifacts)
        factorial_plot(reports["v2"], v2, ["gru", "mamba"], (4096, 16384), "v2", metric, args.out, artifacts)
    real_plots(reports["real"], args.out, artifacts)
    contrast_saved = contrast_plot(reports["v1"], reports["v2"], args.out, artifacts) if args.include_contrasts else False
    for key, path in paths.items():
        shutil.copyfile(path, args.out / f"{key}_analysis.json")
        write_csv(args.out / f"{key}_绘图数据.csv", reports[key]["cells"])
    shutil.copyfile(__file__, args.out / "绘图脚本.py")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "matplotlib_version": matplotlib.__version__,
                **style, "source_analyses": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
                "script_sha256": digest(__file__), "standard_errorbar": "sample standard deviation across 5 training seeds; NOT confidence interval",
                "contrast_plot_saved": contrast_saved, "artifacts": artifacts,
                "limits": ["Standalone PNG300dpi and vector PDF; inspect rendered images for clipping before delivery.",
                           "No synthetic values or missing baselines are substituted.",
                           "Fixed tasks and observed real objects do not establish universal architecture ranking."]}
    (args.out / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": str(args.out), "artifacts": len(artifacts), "contrast_plot_saved": contrast_saved}, ensure_ascii=False))


if __name__ == "__main__":
    main()
