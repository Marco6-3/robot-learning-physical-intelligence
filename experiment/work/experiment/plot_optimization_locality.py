"""Chinese publication figures from the complete, audited 100-run v3 study.

This script only reads saved results/statistics. It does not train, recompute
confidence intervals, or render partial results. Earlier figures are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_results import ROOT, COLORS, configure_style, digest, save_figure, write_csv


VARIANTS = ("gru", "gru_conv1", "gru_conv4", "mamba_conv4", "mamba_conv1")
SAMPLERS = ("uniform", "balanced")
METRICS = ("ap", "query_ap", "cue_shuffled_ap", "cue_shuffled_query_ap")
NAMES = {"gru": "原始 GRU", "gru_conv1": "GRU + conv1", "gru_conv4": "GRU + conv4",
         "mamba_conv4": "Mamba conv4", "mamba_conv1": "Mamba conv1"}
SAMPLER_NAMES = {"uniform": "均匀采样", "balanced": "分层采样"}


def check_interval(row, confidence=None):
    if any(key not in row or not np.isfinite(row[key]) for key in ("estimate", "lower", "upper", "confidence")):
        raise ValueError(f"Missing or nonfinite interval: {row}")
    if row["lower"] > row["upper"] or row.get("undefined_draws", 0):
        raise ValueError(f"Invalid or undefined interval: {row}")
    if confidence is not None and not np.isclose(row["confidence"], confidence, atol=1e-12):
        raise ValueError(f"Unexpected confidence level: {row}")


def load_checked(analysis_path, source):
    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    if report["input_runs"] != 100 or report["task"] != "adaptive-optimization-locality-mechanism-study":
        raise ValueError("The complete fixed 100-run v3 analysis is required")
    for key in ("all_100_results_complete", "exact_test_metadata_reconstruction",
                "all_20_tuning_runs_and_family_selection_verified", "all_sampling_counts_indices_and_visits_replayed",
                "sampler_pairs_have_identical_initial_state_hashes"):
        if report["validation"].get(key) is not True:
            raise ValueError(f"Required independent analysis audit is absent: {key}")
    if report["bootstrap"]["draws"] < 5000:
        raise ValueError("Registered final figures require at least 5000 bootstrap draws")
    if digest(source / "manifest.json") != report["manifest_sha256"]:
        raise ValueError("Analysis belongs to a different result manifest")
    file_checks = {entry["result"]: entry for entry in report["validation"]["files"]}
    if len(file_checks) != 100:
        raise ValueError("Expected 100 unique analyzed result identities")
    rows, lookup = [], {}
    for variant in VARIANTS:
        for sampler in SAMPLERS:
            for seed in range(10):
                stem = f"{variant}_{sampler}_seed{seed}"
                jp, pp, cp = (source / (stem + suffix) for suffix in (".json", "_predictions.npz", ".pt"))
                if any(not path.is_file() for path in (jp, pp, cp)):
                    raise FileNotFoundError(f"Incomplete model/metadata/prediction triplet: {stem}")
                evidence = file_checks[jp.name]
                if digest(jp) != evidence["result_sha256"] or digest(pp) != evidence["predictions_sha256"]:
                    raise ValueError(f"Source changed after independent analysis: {stem}")
                result = json.loads(jp.read_text(encoding="utf-8"))
                if (result["variant"], result["sampler"], result["seed"]) != (variant, sampler, seed):
                    raise ValueError(f"Unexpected result identity: {stem}")
                row = {"variant": variant, "sampler": sampler, "seed": seed,
                       **{metric: float(result["metrics"][metric]) for metric in METRICS}}
                if any(not 0 <= row[metric] <= 1 for metric in METRICS):
                    raise ValueError(f"Invalid probability-ranking metric: {stem}")
                rows.append(row)
                lookup[(variant, sampler, seed)] = row
    cells = {(row["variant"], row["sampler"], row["metric"]): row for row in report["cells"]}
    expected_cells = {(variant, sampler, metric) for variant in VARIANTS for sampler in SAMPLERS for metric in METRICS}
    if set(cells) != expected_cells or len(report["cells"]) != len(expected_cells):
        raise ValueError("Incomplete or duplicate checked metric cells")
    for key, row in cells.items():
        check_interval(row, .95)
        values = [lookup[(key[0], key[1], seed)][key[2]] for seed in range(10)]
        np.testing.assert_allclose(np.mean(values), row["estimate"], atol=1e-7, rtol=0)
    primary = {row["contrast"]: row for row in report["primary_contrasts"] if row["primary"] and row["metric"] == "ap"}
    if set(primary) != {"C1", "C2", "C3"}:
        raise ValueError("The three primary AP contrasts are required")
    for row in primary.values():
        check_interval(row, 1 - .05 / 3)
        if not np.isclose(row["margin"], .03):
            raise ValueError("Unexpected registered meaningful-effect margin")
    test_rows = [row for row in report["data_counts_and_baselines"] if row["split"] == "test"]
    if len(test_rows) != 1:
        raise ValueError("Missing actual test baseline")
    test = test_rows[0]
    if (test["n_records"], test["positive_records"], test["query_records"]) != (65536, 671, 1316):
        raise ValueError("Unexpected fixed test counts")
    return report, rows, lookup, cells, primary, test


def model_color(variant):
    return COLORS["gru"] if variant.startswith("gru") else COLORS["mamba"]


def interval_marker(ax, x, summary, color="#202B36", marker="D", size=26):
    ax.vlines(x, summary["lower"], summary["upper"], color=color, lw=1.65, zorder=6)
    ax.hlines([summary["lower"], summary["upper"]], x-.043, x+.043, color=color, lw=1.1, zorder=6)
    ax.scatter(x, summary["estimate"], color=color, marker=marker, s=size, zorder=7)


def format_probability_axis(ax):
    ax.set_ylim(-.025, 1.045)
    ax.set_xlim(-.26, 1.26)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.65)


def paired_sampler_figure(lookup, cells, test, output, artifacts):
    fig, axes = plt.subplots(1, 5, figsize=(12.4, 4.8), sharey=True)
    offsets = np.linspace(-.055, .055, 10)
    for ax, variant in zip(axes, VARIANTS):
        color = model_color(variant)
        format_probability_axis(ax)
        for seed, offset in enumerate(offsets):
            y = [lookup[(variant, sampler, seed)]["ap"] for sampler in SAMPLERS]
            ax.plot(np.array([0, 1])+offset, y, color=color, alpha=.32, lw=.75, zorder=2)
            ax.scatter(np.array([0, 1])+offset, y, color=color, alpha=.75, s=19, zorder=3)
        for x, sampler in enumerate(SAMPLERS):
            interval_marker(ax, x, cells[(variant, sampler, "ap")])
        ax.axhline(test["query_only_ap"], ls=(0, (4, 3)), lw=.9, color="#9AA3AB", zorder=1)
        ax.set_xticks([0, 1], ["均匀\nuniform", "分层\nbalanced"])
        ax.set_title(NAMES[variant], loc="center", pad=10)
    axes[0].set_ylabel("记录终点 AP")
    title = "稀疏短记忆：改变采样方式后的配对种子表现"
    fig.suptitle(title, x=.06, y=.985, ha="left", fontsize=15, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color="#7C8A96", lw=.8, label="每条连线：同一个训练种子"),
               Line2D([0], [0], marker="D", color="#202B36", lw=1.3, label="均值与 95% 配对重采样区间"),
               Line2D([0], [0], color="#9AA3AB", ls="--", label=f"查询门基线 {test['query_only_ap']:.3f}")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .925), ncol=3, frameon=False)
    fig.text(.06, .018, "每个变体完整展示 10 个训练种子。全体记录 AP；每条记录只评分一个终点。均值区间为辅助描述，三项主效应见专门森林图。\n"
             "两采样方式具有相同的有限池期望 BCE 目标与 400 次更新；分层采样增加已有阳性记录的重复呈现。", fontsize=8.5, color="#4D5863")
    fig.subplots_adjust(left=.06, right=.985, top=.78, bottom=.215, wspace=.20)
    save_figure(fig, output, "v3_采样方式_配对种子", title, artifacts)


def primary_forest(primary, output, artifacts):
    rows = [primary[name] for name in ("C1", "C2", "C3")]
    lower = min(-.05, min(row["lower"] for row in rows))
    upper = max(.08, max(row["upper"] for row in rows))
    padding = max(.018, (upper-lower)*.08)
    fig, ax = plt.subplots(figsize=(10.1, 4.8))
    ax.axvspan(.03, upper+padding, color="#EDF4F6", zorder=0)
    ax.axvline(0, color="#7D8791", lw=1., ls="--", zorder=1)
    ax.axvline(.03, color="#4B8291", lw=1., ls=(0, (2, 2)), zorder=1)
    labels = ["C1  原始 GRU\n分层 − 均匀采样", "C2  GRU · 均匀采样\nconv4 − conv1",
              "C3  Mamba · 均匀采样\nconv4 − conv1"]
    for i, row in enumerate(rows):
        color = COLORS["gru"] if i < 2 else COLORS["mamba"]
        ax.hlines(i, row["lower"], row["upper"], color=color, lw=2.1, zorder=3)
        ax.vlines([row["lower"], row["upper"]], i-.045, i+.045, color=color, lw=1.2, zorder=3)
        ax.scatter(row["estimate"], i, color=color, marker="D", s=38, zorder=4)
        ax.text(1.035, i, f"{row['estimate']:+.3f}\n[{row['lower']:+.3f}, {row['upper']:+.3f}]",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9)
    ax.set_yticks(range(3), labels)
    ax.set_ylim(2.5, -.6)
    ax.set_xlim(lower-padding, upper+padding)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("AP 差值（正值支持对应干预）")
    title = "三项主效应：98.333% 校正区间"
    fig.suptitle(title, x=.05, y=.985, ha="left", fontsize=15, fontweight="bold")
    fig.text(.05, .88, "虚线：零差异；点线：预先固定的实质改善门槛 +0.03；浅色区域：超过门槛。", fontsize=9, color="#4D5863")
    fig.text(.05, .025, "使用 5,000 次配对训练种子与测试记录 bootstrap；图中直接读取分析文件的区间，不另行拟合。\n"
             "三个主对比仅在本阶段内作 Bonferroni 校正。10 个训练种子的百分位区间属近似推断，且以已选择的学习率/检查点为条件。",
             fontsize=8.5, color="#4D5863")
    fig.subplots_adjust(left=.23, right=.80, top=.80, bottom=.23)
    save_figure(fig, output, "v3_三项主效应_校正区间", title, artifacts)


def cue_shuffle_figure(lookup, cells, test, output, artifacts):
    fig, axes = plt.subplots(2, 5, figsize=(12.4, 7.5), sharey=True, sharex=True)
    offsets = np.linspace(-.055, .055, 10)
    for ri, sampler in enumerate(SAMPLERS):
        for ci, variant in enumerate(VARIANTS):
            ax, color = axes[ri, ci], model_color(variant)
            format_probability_axis(ax)
            for seed, offset in enumerate(offsets):
                row = lookup[(variant, sampler, seed)]
                y = [row["query_ap"], row["cue_shuffled_query_ap"]]
                ax.plot(np.array([0, 1])+offset, y, color="#A4ADB5", alpha=.45, lw=.7, zorder=2)
                ax.scatter(offset, y[0], color=color, alpha=.78, s=18, zorder=3)
                ax.scatter(1+offset, y[1], facecolor="white", edgecolor="#737B84", marker="^", s=22, lw=.8, zorder=3)
            interval_marker(ax, 0, cells[(variant, sampler, "query_ap")], color=color)
            interval_marker(ax, 1, cells[(variant, sampler, "cue_shuffled_query_ap")], color="#495661")
            ax.axhline(test["query_only_ap"], color="#9AA3AB", lw=.9, ls=(0, (4, 3)), zorder=1)
            ax.set_xticks([0, 1], ["原始线索", "打乱线索"])
            if ri == 0:
                ax.set_title(NAMES[variant], pad=10)
            if ci == 0:
                ax.set_ylabel(SAMPLER_NAMES[sampler]+"\n查询条件 AP")
    title = "线索打乱诊断：模型是否使用了被标记的比特"
    fig.suptitle(title, x=.06, y=.985, ha="left", fontsize=15, fontweight="bold")
    handles = [Line2D([0], [0], color="#A4ADB5", marker="o", lw=.75, label="同一种子的原始与打乱表现"),
               Line2D([0], [0], color="#202B36", marker="D", label="均值与辅助 95% 区间"),
               Line2D([0], [0], color="#9AA3AB", ls="--", label=f"条件常数基线 {test['query_only_ap']:.3f}")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .943), ncol=3, frameon=False)
    fig.text(.06, .017, "仅在 query=1 的 1,316 条测试记录上计算 AP；每个格完整保留 10 个种子。打乱只改变 cue 标记处的符号，原标签和查询门保留。\n"
             "此图为辅助机制诊断，未加入三项主对比的多重校正；采用固定循环置换，不能据此宣称任意置换下的总体保证。", fontsize=8.5, color="#4D5863")
    fig.subplots_adjust(left=.065, right=.985, top=.845, bottom=.145, hspace=.24, wspace=.19)
    save_figure(fig, output, "v3_线索打乱_查询条件", title, artifacts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "work/results/optimization_locality_v3")
    parser.add_argument("--analysis", type=Path, default=ROOT / "work/results/optimization_locality_v3_analysis/analysis.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/figures_adaptive")
    args = parser.parse_args()
    old_figures = (ROOT / "outputs/figures").resolve()
    output = args.out.resolve()
    if output == old_figures or old_figures in output.parents:
        raise ValueError("Preserve the old figure directory; write adaptive-stage figures separately")
    report, rows, lookup, cells, primary, test = load_checked(args.analysis, args.source)
    style = configure_style()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    paired_sampler_figure(lookup, cells, test, output, artifacts)
    primary_forest(primary, output, artifacts)
    cue_shuffle_figure(lookup, cells, test, output, artifacts)
    write_csv(output / "v3_逐种子绘图数据.csv", rows)
    shutil.copyfile(args.analysis, output / "v3_analysis.json")
    shutil.copyfile(__file__, output / "plot_optimization_locality_source.py")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **style,
                "source_results": str(args.source), "analysis_path": str(args.analysis),
                "analysis_sha256": digest(args.analysis), "script_sha256": digest(__file__),
                "source_run_count": len(rows), "all_required_result_hashes_verified": True,
                "raw_points": "individual paired training seeds; exactly ten per model/sampler cell",
                "primary_errorbars": "98.333-percent paired seed-and-record bootstrap intervals; three-contrast within-stage correction",
                "auxiliary_errorbars": "95-percent paired seed-and-record bootstrap intervals; descriptive",
                "artifacts": artifacts,
                "limits": ["No missing results or baselines are fabricated.",
                           "New figures use a separate directory and unique v3 stems; previous figures are preserved.",
                           "PNG 300dpi and vector PDF; rendered files require visual inspection before delivery."]}
    (output / "optimization_figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "figure_files": len(artifacts), "complete_runs": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
