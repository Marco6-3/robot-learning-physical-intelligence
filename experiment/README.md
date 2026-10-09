# 数据与复现说明

本说明记录已核实的数据、代码来源、文件格式和运行方式。模型结论以最终结果报告及归档预测为准；以下命令本身不表示相应训练已经完成。复现包由实验结束后的快照生成，目标目录为 `复现实验/`，其中保留 `work/` 和 `outputs/` 布局。本文中的相对路径均以该复现包根目录为基准。

## 1. 五套实验各自回答什么问题

| 实验 | 数据与目标 | 不能据此声称 |
|---|---|---|
| `synthetic.py`，v1 固定延迟符号读取 | 阳性率 1%/20% × 延迟 2/64 步 × 64/256 条训练记录；MLP、局部 TCN、GRU、Mamba；预测当前查询对应的历史符号 | 物理接触仿真；纯粹的单状态记忆时长效应。长延迟还增加随机历史容量要求 |
| `real_experiment.py`，真实滑移检测 | uSkin 逐试验人工标签；18/54 条训练试验及留一物体；因果输入检测当前滑移 | 1% 稀疏事件验证、必要记忆长度的因果识别、提前预测、闭环恢复成功 |
| `cue_memory.py`，v2 单状态记忆补充 | 每记录仅一个相关 cue 位；阳性率 1%/20% × 保持 2/64 步 × 4,096/16,384 条训练记录；GRU、Mamba | 真实接触任务的控制收益；对 v1 负结果的替换 |
| `optimization_locality.py`，v3 采样与局部卷积干预 | 固定 1% 稀疏、2 步延迟、4,096 条训练记录；5 个模型变体 × 2 种目标期望相同的采样方式；10 个新种子 | 长记忆、真实触觉和闭环收益；把新机制发现追写为最初盲法假设 |
| `retention_curriculum.py`，长记忆训练路径 | 固定 20% 阳性、1 bit；GRU/Mamba 各比较直接训练 lag64 和渐进 lag2/4/8/16/32/64；每次连续 2,400 次更新 | 同等固定 lag64 任务曝光下的架构比较；失败就证明内在记忆容量不足 |

v2 单独登记、单独归档。其相关记忆容量固定为 1 bit，但训练规模与 v1 不同，不能直接把两套任务的绝对分数当作同一学习曲线。协议原文与登记哈希位于 `outputs/实验协议.md`、`outputs/补充协议_单状态记忆.md` 和对应登记文件。

v3 是看到旧结果之后的自适应后续研究。其协议 `outputs/补充协议_采样与局部卷积.md` 和代码依赖于 2026-10-09 20:30:37（Asia/Shanghai）封存，新阶段使用独立生成的数据：训练种子 510000–510009，调参训练 510901，验证 520001 共 8,192 条，测试 530001 共 65,536 条。新测试含 671 个阳性、1,316 个查询；这只是生成器计数，不是模型结果。

v3 的五个变体为原始 GRU、GRU conv1+SiLU、GRU conv4+SiLU、Mamba conv4、Mamba conv1，分别有 19,169、19,233、19,329、19,449、19,041 个可训练参数。公共形状参数显式配对，不同形状的核保留自身默认初始化。uniform 使用实际训练阳性比例 pi 对 BCE 加权，balanced 每批正负各 16 条并将无权 BCE 乘 2(1-pi)；二者期望目标相同，但正例曝光、梯度裁剪和 Adam 更新不同。每个家族按全部变体/采样方式的平均验证 AP 共选一个学习率，20 次调参后固定运行 100 次最终训练。不得把相同训练池与更新次数称作相同正例呈现预算。

三项 v3 主对比为 GRU 的 balanced−uniform、uniform 下 GRU conv4−conv1、uniform 下 Mamba conv4−conv1；配对种子和测试记录 bootstrap 5,000 次，分别使用 98.333% 区间与 AP 0.03 门槛。辅助恢复判据要求 balanced 原始 GRU 的平均 AP 95% 下界超过 0.90，同时检查查询条件表现和线索打乱。打乱保留查询信号，参照 AP 为约 0.51，不能误用总体阳性率约 0.01。

第五套训练路径研究也是旧现象可见后的新机制实验，于 2026-10-09 20:28:32（Asia/Shanghai）封存，见 `outputs/补充协议_记忆训练路径.md` 及登记文件。训练生成种子 610000–610004，每池 4,096 条；调参训练 610901；验证 620001 共 4,096 条；测试 630001 共 16,384 条。新测试实际含 3,352 个阳性、6,561 个查询。两条路径共享标签、记录及阳性呈现次序；不同 lag 只改变相关线索的位置，不提供六份新标签。

direct 在 lag64 连续训练 2,400 步；curriculum 依次用 lag2、4、8、16、32、64 各 400 步。batch32，共 76,800 次记录呈现；每次运行保留同一个模型、同一个 AdamW 及其动量，阶段间不重置。每 100 步都在固定 lag64 验证集评估，按验证全记录 AP 选择检查点。每模型的两条路径共享一个学习率；8 次调参后运行 20 次正式训练（2 模型 × 2 路径 × 5 种子）。最终测试只评估最佳验证检查点，保存的最后一步权重不能被误称为已测过的最终测试表现。

训练路径分析以两个模型各自的 `curriculum−direct` 查询条件 AP 为两个主对比，5,000 次配对种子/记录 bootstrap，使用 97.5% 区间；架构差距及交互为描述性。辅助成功标准是单次 query AP>0.90 且打乱线索后 query AP 下降>0.30，只有 5/5 种子都满足才称为这五个种子上的稳定成功。curriculum 只有最后 400 步直接训练 lag64，direct 则有 2,400 步，故不能称为同等固定长任务曝光。两者若都失败，仍不能推断内在记忆容量上限。

训练路径的分析读取校验有一份明确保留的数值勘误：个别 sklearn 验证 AP 因双精度舍入成为 1+2e−16 或 1+4e−16，原先严格的 [0,1] 检查因此拒绝正常记录。仅将这一合法性容差放宽到 [−1e−12,1+1e−12]，不裁剪原值，也不改变训练、模型选择、指标或统计规则。原分析源码、原登记和旧/新 SHA 均保留，详见 `outputs/记忆训练路径_浮点校验勘误.md` 及配套 JSON；不能把更正后的分析源码说成完全未经修改。

## 2. 真实数据来源、许可与校验

数据来自作者官方仓库 [ARQ-CRISP/slip_detection_dataset_2021](https://github.com/ARQ-CRISP/slip_detection_dataset_2021)，对应 [IROS 2021 论文](https://doi.org/10.1109/IROS51168.2021.9636602)，采用 BSD-3-Clause。作者许可证原文随包保留在 `work/real_data/sources/arq2021_LICENSE`，再分发时应保留该许可及作者声明。

固定提交为 `a536ee3301af0566218782b00de334b6a6630979`。已下载三个官方 Git LFS 文件，共 **266,271,772 字节**。本次实际 SHA-256 与对应固定提交的 LFS 指针一致：

| 原始文件 | 字节数 | SHA-256 |
|---|---:|---|
| `slipDataset_brush_tactile.h5` | 41,105,856 | `a2aef56bf4cfc5e2d17ae09772295cb207c733e3a73059913939ba5a7a1e2139` |
| `slipDataset_screwDriver_tactile.h5` | 82,416,420 | `c8124b295a6faf42c750c75e8dcd49dcc45289e06fa7cb4047476af2f570309a` |
| `slipDataset_SpoolSolder_tactile.h5` | 142,749,496 | `a8c4f14e71c0b4d7da738997c965e745f75123fea7f5c6d338810ba4adfdeeff` |

每个来源 URL、文档哈希和验证状态均记录于 `work/real_data/sources/source_manifest.json`。复现包包含导出的逐试验 NPZ、审计、来源文档和许可，**不重复包含原始 HDF5**；原始文件可按固定版本重新获取。

### 实际审计的规模和标签

共 90 次完整试验，3 个物体 × 3 个姿态 × 10 次重复；228,521 帧，累计时间戳跨度 **1,269.12 秒**。每帧 18 个触点 × XYZ 三方向，共 54 通道。读数是原始霍尔传感器值，不是已标定的牛顿值。

作者声明频率为 180 Hz；实际逐试验时间戳严格递增，原单位为毫秒，间隔中位数 5.546 ms，最大间隔 20.63 ms。导出转为秒，未用名义频率替代实际时间。

| 人工标签 | 解释 | 帧数 |
|---|---|---:|
| 0 | 未持物，或持物且机械臂静止；未观察到滑动 | 95,623 |
| 1 | 机械臂静止时滑动 | 24,436 |
| 2 | 释放物体 | 1,938 |
| 3 | 抓取物体 | 9,381 |
| 4 | 机械臂运动时未滑动 | 61,365 |
| 5 | 机械臂运动时滑动 | 34,835 |
| 6 | 其他触觉事件；二分类评分中排除 | 943 |

实验使用 `tactile_slips_label`，将 1/5 作为滑移阳性。没有把由输入变化规则产生的 `tactile_changes_label` 当作真值。按有效非滑移到滑移的原始帧转换计算，共 238 个起点；记录首帧已经滑移的情况不算新起点。**作者提供人工滑移标签；标注观察来源及时间精度未核实**。因此不将标签称作已验证的独立物理真值或无误差测量。此前协议关于独立标签的表述属于尚未核实的筹备假定，由 `标签来源补充审计.md` 明确更正；冻结协议原文保留。

标签 0 混合未接触和稳定接触，无法据此唯一恢复二元接触起点。只有 3 个物体和 90 个独立试验，不能把 22.9 万帧当作同等数量的独立实验。26 次试验短于 5 秒；模型输入的历史长度不等同于任务真正需要的记忆长度。

### 实验用数据处理

`work/real_data/preprocess.py` 将每次试验导出为 `processed/<物体>_Pose<姿态>_Exp<编号>.npz`，保存 `x(T,54)`、原标签 `label`、`timestamp_s`、二分类 `slip`、`valid` 和 `onset`。原读数转成 float32 后已逐值核对，没有数值损失。90 个原始试验数组均不同。预处理没有对整套数据拟合归一化或滤波器。

真实实验另取原始第 5、11、17……帧为端点，以端点及之前 5 帧的均值作为输入，名义为 30 Hz；标签取端点标签，6 个原始标签均有效才评分。共得到 38,050 个端点，其中 37,881 个有效，阳性占 **26.09%**，因此是自然中等事件密度的外部证据。

先按完整试验划分，再生成窗口。主划分为每个物体/姿态的 Exp1–6 训练、Exp7–8 验证、Exp9–10 测试，即 54/18/18 次；小训练集只用 Exp1–2，共 18 次。另保留三个留一物体划分。归一化只拟合当前训练集。长度 128 的输入窗口使用 64 步上下文，后 64 步不重叠评分；起始补零发生在归一化后，补齐位置不评分，每个真实降采样端点恰评分一次。

完整记录见 `work/real_data/arq2021_audit.json`、`trials.csv`、`splits.json` 和 `work/results/real/chunk_validation.json`。

## 3. 为什么没有把 FlexiTac 现成版本当作滑移真值

已实时核对 [Tna001/tactile_test_tube_pyflexitac](https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac)，数据卡标注 Apache-2.0；固定版本为 `ebd3c711d678aa18bfc7f6a0a1e894104c8c1ea4`。来源卡和 `meta/info.json` 已归档，当前版本与本机原有缓存一致。本次仅只读检查原项目缓存，没有修改它。

已检查全部 10 个 parquet：64 回合、75,041 帧、30 Hz；实际字段包含 12×32 触觉、动作、关节状态、时间和索引，没有独立接触、滑移、失败或成功标签。逐文件哈希与字段清单位于 `work/real_data/flexitac_label_audit.json`。

对触觉输入阈值化产生标签只是在检验模型恢复这一规则；未来动作回归也会改变研究问题，不能代替独立滑移验证。公开视频可能支持另行人工标注，但本轮未做。30 Hz 的记录每 100 ms 约 3 帧，也不能证明原生传感器的端到端响应延迟。复现包不包含这个独立的既有缓存，只保留审计及来源证据。

## 4. 模型来源与运行环境

便携 Mamba-1 依据 [state-spaces/mamba](https://github.com/state-spaces/mamba) 的固定提交 `e9594ce1c732d97440f0332fdc43170a2294dbfa` 实现；原代码和 Apache-2.0 许可保存在 `work/references/`。`mamba_source_manifest.json` 记录了源码 URL、提交和每个文件 SHA-256。模型保留输入依赖 Δ/B/C、A/D、因果深度卷积、门控及特殊初始化，使用纯 PyTorch 非融合扫描，不要求安装 `mamba-ssm` 或编译其 CUDA 扩展。

本机已核实的环境为 Python **3.11.9**、PyTorch **2.11.0+cu128**、CUDA runtime **12.8**、NVIDIA GeForce RTX 5060 Laptop GPU；其余版本在根目录 `requirements.txt` 中固定。硬件/软件完整快照保存在冻结登记及运行 manifest。代码可以选择 CPU，但 CPU 重跑全部训练的耗时没有在本轮验证；跨设备或版本不保证逐位相同。

先建立独立 Python 3.11 环境，再安装依赖及适合硬件的 PyTorch。PyTorch 的 CPU/CUDA 构建应按[官方安装页面](https://pytorch.org/get-started/locally/)选择；若希望尽量接近本次运行，应使用记录中的 2.11.0、cu128 构建。

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`requirements.txt` 不自动选择 PyTorch 的硬件构建。数值检查需要 einops；数据重新读取需要 h5py，FlexiTac 可选审计需要 pyarrow。其余模型训练使用 numpy、scikit-learn 和 PyTorch。网络仅用于初次安装与可选原始数据下载；包内逐试验数据和合成生成器足以离线运行训练。

静态论文图使用 Matplotlib 3.10.7，依赖版本已纳入 requirements。中文图沿用 Windows 微软雅黑字体 `C:/Windows/Fonts/msyh.ttc`；其他系统需提供该字体或在 `plot_results.py` 的 `configure_style()` 中明确指定可用中文字体。字体变更影响图形外观，不改变保留的统计值。

## 5. 先校验，再查看或复算

复现包生成时，`manifest.json` 列出文件大小和 SHA-256；`SHA256SUMS.txt` 另包含 manifest 的 SHA-256。校验清单不能递归计算自己的哈希，因此不把自身列为自校验项。生成过程对源文件和复制文件逐个核对，并检查复制期间源目录是否改变。

在复现包根目录执行：

```powershell
python work/package_results.py --verify
```

此命令只读检查归档。训练命令可能断点续跑或写入新结果，环境变更也可能触发真实实验的指纹保护。为了保留交付结果，完整重训应建立一个不含旧 `work/results/` 的新工作副本；已有复现目录不会被自动清空。下面示例要求相邻的 `reproduction_rerun` 尚不存在：

```powershell
if (Test-Path -LiteralPath ../reproduction_rerun) { throw 'reproduction_rerun already exists; choose a new directory' }
New-Item -ItemType Directory -Path ../reproduction_rerun/work -ErrorAction Stop
Copy-Item -LiteralPath ./work/experiment,./work/real_data,./work/references,./work/frozen -Destination ../reproduction_rerun/work -Recurse
Copy-Item -LiteralPath ./outputs -Destination ../reproduction_rerun -Recurse
Copy-Item -LiteralPath ./requirements.txt -Destination ../reproduction_rerun
Set-Location ../reproduction_rerun
```

目录名可更换。新副本保留原来的协议登记；不要重新运行 `freeze.py` 或覆盖各补充研究的登记。重训得到的新结果应与已交付结果分开解释。

### 可选：从官方原始数据重做逐试验导出

归档已经包含 `processed/*.npz`，无需再次下载。若要逐级复核数据处理：

```powershell
python work/real_data/acquire.py --download
python work/real_data/preprocess.py --skip-flexitac
```

下载器使用归档提交和 LFS 哈希，ARQ 原始下载上限为 350 MB；默认沿用归档 FlexiTac 版本。`--skip-flexitac` 仅跳过依赖另一个独立缓存的本地审计，保留已归档审计。若另有完整 FlexiTac 缓存，可用 `--flexitac-cache <缓存目录>`，其中应包含 parquet、`revision.txt` 和 `all_data.npz`。

### 数值与因果检查

```powershell
python work/experiment/test_models.py --device cpu
python work/experiment/validate_protocol.py
python work/experiment/validate_final_configs.py
python work/experiment/analyze_synthetic.py --self-test
python work/experiment/analyze_real.py --self-test
python work/experiment/analyze_cue_memory.py --self-test
python work/experiment/analyze_optimization_locality.py --self-test
python work/experiment/analyze_retention_curriculum.py --self-test
```

这些命令检查模型与分析实现，不能替代正式实验结果。三个补充研究脚本的 `--precheck` 都会写入不可覆盖的新阶段登记；交付包已经保留其登记，不要在同一副本重复执行。v3 已执行的 CPU 前检记录在 `work/experiment/optimization_locality_precheck.json`，核对了因果性、梯度有限、公共权重配对和经验目标等价；长记忆训练路径的前检另存为 `retention_curriculum_precheck.json`。

### 五套正式实验命令

以下命令从新的工作副本根目录顺序执行，不建议在同一块 GPU 上同时运行多个训练进程。

```powershell
python work/experiment/synthetic.py --out work/results/synthetic --steps 400
python work/experiment/analyze_synthetic.py --source work/results/synthetic --out work/results/synthetic_analysis

python work/experiment/real_experiment.py --prepare-only
python work/experiment/real_experiment.py --suite held_trial
python work/experiment/analyze_real.py --suite held_trial
python work/experiment/real_experiment.py --suite loo
python work/experiment/analyze_real.py --suite all

python work/experiment/cue_memory.py --run --out work/results/cue_memory_v2
python work/experiment/analyze_cue_memory.py --source work/results/cue_memory_v2 --out work/results/cue_memory_v2_analysis

python work/experiment/optimization_locality.py --run --out work/results/optimization_locality_v3
python work/experiment/analyze_optimization_locality.py --source work/results/optimization_locality_v3 --out work/results/optimization_locality_v3_analysis --draws 5000 --margin 0.03 --rescue-ap 0.90

python work/experiment/retention_curriculum.py --run --out work/results/retention_curriculum
python work/experiment/analyze_retention_curriculum.py --source work/results/retention_curriculum --out work/results/retention_curriculum_analysis --draws 5000
```

学习率、种子、更新次数、训练规模、主指标及统计阈值应保持登记配置。不要用 `--smoke` 的输出替代正式实验。v2/v3 结果格式为每记录单个标量，不能直接交给预期 v1 序列预测格式的分析函数。v3 分析要求完整的 100 个最终结果、20 个调参记录和对应权重/预测，独立重建测试元数据及采样索引；不完整时会停止而非汇报部分优胜结果。

### 完整结果的静态图

v3 绘图需完整的 100 次正式运行及上述独立分析 JSON；绘图时会再次核对源结果/预测哈希，读取已计算的区间，不重新拟合统计。下面命令生成配对种子图、三项主效应森林图、查询条件线索打乱图，均为中文 PNG 300 dpi 与矢量 PDF：

```powershell
python work/experiment/plot_optimization_locality.py --source work/results/optimization_locality_v3 --analysis work/results/optimization_locality_v3_analysis/analysis.json --out outputs/figures_adaptive
```

新图保存在 `outputs/figures_adaptive/`，不会覆盖旧的 `outputs/figures/`；输出包括绘图源数据 CSV、分析副本和图形哈希清单。正式交付前还需对实际渲染结果进行视觉检查。此命令列出不代表图形已经生成。

## 6. 模型、预测与诊断文件在哪里

| 位置 | 内容 |
|---|---|
| `work/frozen/` | 早期协议和执行源码的不可覆盖快照；后续独立补充以各自登记为准 |
| `work/results/synthetic/` | 每个条件与种子的 `.json`、`.pt`、`_predictions.npz`，以及学习率调参记录；预测含 y/p/query 和打乱历史符号后的概率 |
| `work/results/synthetic_analysis/` | 从保留预测重新计算的统计与诊断 |
| `work/results/real/pretraining_manifest.json` | 真实实验代码、数据和配置指纹；`chunk_validation.json` 检查评分覆盖 |
| `work/results/real/<配置>/<模型>/<运行>/` | `result.json`、`checkpoint.pt`；`test_predictions/<trial>.npz` 保存无 padding 的 p/y/valid/timestamp_s/coverage_s/source_endpoint_index；正式运行另含验证集预测 |
| `work/results/real_analysis/` | 试验与种子配对不确定性、逐试验复核、事件指标和持续报警诊断 |
| `work/results/cue_memory_v2/` | v2 的登记、调参、逐条件种子结果、权重和标量预测；预测含 y/p/query/cue_shuffled_p/endpoint/cue/bit |
| `work/results/cue_memory_v2_analysis/` | v2 的配对统计、精确数据基线、线索消融和完整 bootstrap 向量 |
| `work/results/optimization_locality_v3/` | v3 的 manifest、20 次调参、family_lr_selection.json；100 组 `<variant>_<sampler>_seed<seed>.json/.pt/_predictions.npz`；JSON 另记录阳性曝光、访问数、批次索引和初始权重摘要 |
| `work/results/optimization_locality_v3_analysis/` | 独立重算的逐种子指标、三主对比、辅助等效/恢复/线索消融诊断、采样完整性核对和 bootstrap 向量 |
| `work/results/retention_curriculum/` | 模型级 `selection.json`；各 `<模型>/<direct或curriculum>/<调参或正式运行>/` 内 result.json、training_trace.json、best.pt、final.pt、final_optimizer.pt；正式运行另存测试和验证预测 |
| `work/results/retention_curriculum_analysis/` | 逐种子/条件指标、两条训练路径的主效应、描述性架构/交互、24 点验证曲线和 bootstrap 向量 |
| `outputs/figures_adaptive/` | 后续机制研究的独立图形目录；v3 三图各有 PNG/PDF、绘图源数据、分析副本和哈希清单 |
| `work/optimization_review.md` | v3 的独立方法与实现审查；不替代已封存协议 |
| `work/references/`、`work/real_data/sources/` | 上游代码、数据说明、许可证、固定提交和来源哈希 |

真实检测的事件指标按有效连续区间进行一对一重叠匹配，延迟只针对检出的事件；真实时间分母来自实际时间戳。持续常亮警报可能在单一事件上得到很高的 Event-F1，因此应同时看逐帧 AP、阴性时间报警占比、常亮基线及左删失起点的处理。不能只摘选 Event-F1 或检出事件的平均延迟。

训练与预测计时、显存是本次软件实现的测量。便携 Mamba 使用非融合扫描，而 GRU 可使用 cuDNN，两者耗时差不能直接推论为官方融合实现或架构固有速度差。离线检测也没有提供真机恢复、控制成功率或端到端系统延迟证据。
