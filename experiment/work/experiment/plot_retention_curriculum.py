"""Chinese retention-study figures from completed analyses, never from models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("gru", "mamba")
ARMS = ("direct", "curriculum")
LABELS = {"gru": "GRU", "mamba": "Mamba-1", "direct": "直接训练", "curriculum": "课程训练"}
COLORS = {"gru": "#267EA5", "mamba": "#BB505D", "direct": "#6F7D89"}
STEPS = list(range(100, 2401, 100))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure():
    path = Path("C:/Windows/Fonts/msyh.ttc")
    if not path.is_file():
        path = Path(font_manager.findfont("Microsoft YaHei", fallback_to_default=False))
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#87929C", "axes.linewidth": .65,
                         "grid.color": "#E0E5E9", "grid.linewidth": .55,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    return {"font": name, "font_path": str(path)}


def check_inputs(source):
    required = [source/name for name in ("analysis.json", "per_seed_metrics.csv",
                                         "validation_curves_per_seed.csv", "validation_curves_summary.csv")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Complete analysis and curve CSVs required: " + "; ".join(missing))
    report = json.loads(required[0].read_text(encoding="utf-8"))
    seeds, curves, summaries = (load_csv(path) for path in required[1:])
    if report.get("completed_final_runs") != 20 or report.get("completed_tuning_runs") != 8:
        raise ValueError("Only the complete registered28-run study may be plotted")
    if report["bootstrap"]["draws"] != 5000 or report["bootstrap"]["primary_marginal_confidence"] != .975:
        raise ValueError("Primary figure requires the registered5000draw/97.5% analysis")
    expected = {(model, arm, seed) for model in MODELS for arm in ARMS for seed in range(5)}
    lookup = {(row["model"], row["arm"], int(row["seed"])): row for row in seeds}
    if set(lookup) != expected or len(seeds) != 20:
        raise ValueError("Missing/duplicate paired final-seed rows")
    curve_lookup = {(r["model"], r["arm"], int(r["seed"]), int(r["step"])): r for r in curves}
    expected_curves = {(*cell, step) for cell in expected for step in STEPS}
    if set(curve_lookup) != expected_curves or len(curves) != 480:
        raise ValueError("All480model/path/seed/step curve rows are required")
    if any(int(row["validation_lag"]) != 64 for row in curves):
        raise ValueError("Validation lag must remain64 throughout")
    summary_lookup = {(r["model"], r["arm"], int(r["step"])): r for r in summaries}
    if len(summary_lookup) != 96 or len(summaries) != 96:
        raise ValueError("Expected96 complete validation-curve summaries")
    for model in MODELS:
        for arm in ARMS:
            for step in STEPS:
                row = summary_lookup[(model, arm, step)]
                for metric in ("val_ap", "val_query_ap"):
                    values = np.array([float(curve_lookup[(model, arm, seed, step)][metric]) for seed in range(5)])
                    np.testing.assert_allclose(float(row[metric+"_mean"]), values.mean(), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(float(row[metric+"_seed_sd"]), values.std(ddof=1), rtol=0, atol=1e-12)
    return report, lookup, curve_lookup, summary_lookup, required


def save(fig, output, name, title, artifacts):
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        options = {"dpi": 300} if extension == "png" else {"metadata": {"Title": title, "Creator": "Source-backed Matplotlib research figures"}}
        fig.savefig(path, bbox_inches="tight", pad_inches=.14, **options)
        artifacts.append({"path": path.name, "sha256": digest(path)})
    plt.close(fig)


def paired_figure(report, rows, output, artifacts):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    baseline = report["deterministic_baseline"]["query_constant_score_ap"]
    jitter = np.linspace(-.10, .10, 5)
    for ax, model in zip(axes, MODELS):
        color = COLORS[model]
        for seed, offset in enumerate(jitter):
            for start, metric, linecolor, style in ((0, "query_ap", color, "-"),
                                                    (3, "cue_shuffled_query_ap", "#8C989F", "--")):
                values = [float(rows[(model, arm, seed)][metric]) for arm in ARMS]
                ax.plot(np.array([start, start+1])+offset, values, style, color=linecolor,
                        marker="o", ms=4.2, lw=.95, alpha=.58, zorder=3)
        for start, metric, color_mean in ((0, "query_ap", color), (3, "cue_shuffled_query_ap", "#52626E")):
            for ai, arm in enumerate(ARMS):
                value = np.mean([float(rows[(model, arm, seed)][metric]) for seed in range(5)])
                ax.scatter(start+ai, value, marker="D", s=57, color=color_mean, edgecolor="white", linewidth=.8, zorder=6)
        ax.axhline(baseline, color="#A8B0B7", ls=(0, (2, 3)), lw=1)
        ax.axhline(.90, color="#99B8A5", ls=(0, (6, 3)), lw=.9)
        ax.set_xticks([0, 1, 3, 4], ["直接\n原输入", "课程\n原输入", "直接\ncue 位打乱", "课程\ncue 位打乱"])
        ax.set_xlim(-.42, 4.42)
        ax.set_ylim(-.02, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", zorder=0)
        ax.set_title(LABELS[model], loc="left", fontweight="bold")
        if model == "gru":
            ax.set_ylabel("lag64 测试 query AUPRC")
    title = "长记忆测试：训练路径与 cue 内容消融"
    fig.suptitle(title, x=.075, y=.995, ha="left", fontsize=15, fontweight="bold")
    handles = [Line2D([0], [0], color="#54778C", marker="o", lw=1, label="细线：同一训练种子的配对"),
               Line2D([0], [0], color="#304F63", marker="D", ls="none", label="菱形：5 种子均值"),
               Line2D([0], [0], color="#A8B0B7", ls=":", label=f"常数分数基线 {baseline:.3f}")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .936), ncol=3, frameon=False)
    fig.text(.075, .018, "仅展示由 lag64 验证 AUPRC 选出的最佳检查点；每条测试记录仅评分一个终点。\n"
             "绿色线为辅助阈值 0.90；成功还要求 cue 打乱后的 query AUPRC 下降 >0.30。稳定成功需五个种子全部达标。",
             fontsize=8.5, color="#4D5D68")
    fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.22, wspace=.18)
    save(fig, output, "paired_query_ap_and_shuffle", title, artifacts)


def validation_figure(curves, summaries, output, artifacts):
    fig = plt.figure(figsize=(10.4, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[.24, 1], hspace=.13, wspace=.18)
    for col, model in enumerate(MODELS):
        schedule = fig.add_subplot(grid[0, col])
        ax = fig.add_subplot(grid[1, col], sharex=schedule)
        edges = np.arange(0, 2401, 400)
        schedule.step(edges, [2, 4, 8, 16, 32, 64, 64], where="post", color=COLORS[model], lw=1.7)
        schedule.plot([0, 2400], [64, 64], color=COLORS["direct"], ls="--", lw=1.05)
        schedule.set_yscale("log", base=2)
        schedule.set_yticks([2, 8, 32, 64], ["2", "8", "32", "64"])
        schedule.set_ylim(1.6, 82)
        schedule.set_title(LABELS[model], loc="left", fontweight="bold", pad=8)
        schedule.tick_params(axis="x", labelbottom=False, length=0)
        if col == 0:
            schedule.set_ylabel("训练 lag", fontsize=9)
        for ai, arm in enumerate(ARMS):
            color = COLORS["direct"] if arm == "direct" else COLORS[model]
            for seed in range(5):
                values = [float(curves[(model, arm, seed, step)]["val_query_ap"]) for step in STEPS]
                ax.plot(STEPS, values, color=color, lw=.55, alpha=.15, zorder=1)
            mean = np.array([float(summaries[(model, arm, step)]["val_query_ap_mean"]) for step in STEPS])
            sd = np.array([float(summaries[(model, arm, step)]["val_query_ap_seed_sd"]) for step in STEPS])
            ax.fill_between(STEPS, mean-sd, mean+sd, color=color, alpha=.13, linewidth=0, zorder=2)
            ax.plot(STEPS, mean, color=color, lw=1.7, ls="--" if arm == "direct" else "-",
                    label=LABELS[arm], zorder=3)
        for edge in edges[1:-1]:
            schedule.axvline(edge, color="#DCE2E7", lw=.55, zorder=0)
            ax.axvline(edge, color="#DCE2E7", lw=.55, zorder=0)
        ax.set_xticks(edges, ["0", "400", "800", "1,200", "1,600", "2,000", "2,400"])
        ax.tick_params(axis="x", labelrotation=25)
        ax.set_xlim(0, 2400)
        vals = [float(summaries[(model, arm, step)]["val_query_ap_mean"])+sign*float(summaries[(model, arm, step)]["val_query_ap_seed_sd"])
                for arm in ARMS for step in STEPS for sign in (-1, 1)]
        ax.set_ylim(min(-.02, min(vals)-.02), max(1.02, max(vals)+.02))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", zorder=0)
        ax.set_xlabel("累计参数更新步数")
        if col == 0:
            ax.set_ylabel("验证 query AUPRC（始终 lag64）")
        ax.legend(loc="lower right", frameon=False)
    title = "训练跨度逐级变化；验证任务始终为 lag64"
    fig.suptitle(title, x=.075, y=.995, ha="left", fontsize=15, fontweight="bold")
    fig.text(.075, .015, "上排只表示训练 lag：直接路径始终 64，课程路径每 400 步切换一次。下排全部为同一 lag64 验证集。\n"
             "粗线：5 种子均值；阴影：5 种子的标准差（非置信区间）；浅细线：单种子。两个路径共享学习率，AdamW 连续更新不重置。",
             fontsize=8.5, color="#4D5D68")
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.17)
    save(fig, output, "validation_curve_and_training_lag", title, artifacts)


def primary_effect_figure(report, output, artifacts):
    rows = report["primary_path_effects"]
    by_model = {row["model"]: row for row in rows}
    if len(rows) != 2 or set(by_model) != set(MODELS):
        raise ValueError("Expected exactly the two registered primary path effects")
    if any(row["metric"] != "query_ap" or row["confidence"] != .975 or row["undefined_draws"] != 0 for row in rows):
        raise ValueError("Wrong or undefined primary interval; do not silently draw a replacement")
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    bounds = []
    for y, model in enumerate(MODELS):
        row = by_model[model]
        lo, hi, point = row["lower"], row["upper"], row["estimate"]
        ax.hlines(y, lo, hi, lw=2.3, color=COLORS[model])
        ax.vlines([lo, hi], y-.065, y+.065, lw=1.15, color=COLORS[model])
        ax.scatter(point, y, color=COLORS[model], marker="D", s=48, zorder=4)
        ax.text(.985, y+.16, f"{point:+.3f}  [{lo:+.3f}, {hi:+.3f}]", transform=ax.get_yaxis_transform(),
                ha="right", va="top", fontsize=9, color="#4F5D67")
        bounds.extend([lo, hi, point])
    margin = max(.025, (max(bounds)-min(bounds))*.12)
    ax.set_xlim(min(-.025, min(bounds)-margin), max(.025, max(bounds)+margin))
    ax.axvline(0, color="#9DA8B1", ls="--", lw=.9)
    ax.set_yticks([0, 1], [LABELS[m] for m in MODELS])
    ax.set_ylim(1.48, -.48)
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("lag64 测试 query AUPRC 差值（课程 − 直接）")
    title = "两项主效应：训练路径差值与 97.5% 区间"
    fig.suptitle(title, x=.095, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.095, .015, "配对训练种子＋独立测试记录 bootstrap 5,000 次。两项比较各 97.5%，Bonferroni 名义家族覆盖率 95%。\n"
             "覆盖率近似；区间条件于固定数据划分、共享学习率和检查点选择。这里不检验架构普遍优劣。",
             fontsize=8.5, color="#4D5D68")
    fig.subplots_adjust(left=.115, right=.985, top=.80, bottom=.25)
    save(fig, output, "primary_training_path_effects", title, artifacts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT/"work/results/retention_curriculum_analysis")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/figures_retention")
    args = parser.parse_args()
    report, seeds, curves, summaries, files = check_inputs(args.source)
    style = configure()
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = []
    paired_figure(report, seeds, args.out, artifacts)
    validation_figure(curves, summaries, args.out, artifacts)
    primary_effect_figure(report, args.out, artifacts)
    for path in files:
        shutil.copyfile(path, args.out/path.name)
    shutil.copyfile(__file__, args.out/"绘图脚本.py")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "matplotlib_version": matplotlib.__version__,
                **style, "script_sha256": digest(__file__),
                "sources": [{"path": str(path), "sha256": digest(path)} for path in files],
                "artifacts": artifacts, "curve_error_band": "5 training-seed standard deviations, not confidence intervals",
                "primary_intervals": "source-computed paired bootstrap5000; two97.5% query-AP path effects",
                "all_validation_points_lag": 64,
                "checkpoint_test_scope": "best validation checkpoint only; final validation is not final test",
                "visual_review_required": "Inspect PNG rendering for clipping/overlap before final delivery."}
    (args.out/"figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "figures": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
