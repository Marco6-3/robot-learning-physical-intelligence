# 真实触觉数据：获取、逐试验导出与标签审计

本目录仅负责数据获取和审计，没有训练模型，也没有根据实验效果选择数据划分。原始数据和第三方许可证保存在 `raw/`、`sources/`。

## 数据来源与固定版本

- 来源：[ARQ-CRISP/slip_detection_dataset_2021](https://github.com/ARQ-CRISP/slip_detection_dataset_2021)，BSD-3-Clause。
- 固定提交：`a536ee3301af0566218782b00de334b6a6630979`。
- 三个 HDF5 文件共 266,271,772 字节；实际 SHA-256 均与固定提交的 Git LFS 指针一致，记录在 `sources/source_manifest.json`。
- 论文：Zenha, Denoun, Coppola, Jamone, “Tactile Slip Detection in the Wild Leveraging Distributed Sensing of both Normal and Shear Forces,” IROS 2021，DOI [10.1109/IROS51168.2021.9636602](https://doi.org/10.1109/IROS51168.2021.9636602)。

## 实际读取结果

90 次独立试验，3 个物体 × 3 个姿态 × 10 次重复；228,521 帧，累计 1,269.12 秒。传感器为 18 个触点、每点 3 个方向，共 54 通道。数据是原生霍尔传感器读数，不能直接解释为牛顿。

作者声明采样率为 180 Hz；实际时间戳单位为毫秒，逐试验均严格递增，间隔中位数为 5.546 ms。导出统一使用秒。只有一个间隔超过 20 ms，最大值为 20.63 ms。

使用作者提供的逐帧人工 `tactile_slips_label`。`tactile_changes_label` 来自触觉输入变化的规则，未作为真值导出。**作者提供人工滑移标签；标注观察来源及时间精度未核实**。因此不能把人工标签称作已验证的独立物理真值或无误差测量。此前协议关于独立标签的表述属于尚未核实的筹备假定，由 `outputs/标签来源补充审计.md` 更正；冻结协议原文保留。

| 标签 | 含义 | 帧数 |
|---|---|---:|
| 0 | 未持物，或持物且机械臂静止；未观察到滑动 | 95,623 |
| 1 | 机械臂静止时滑动 | 24,436 |
| 2 | 释放物体 | 1,938 |
| 3 | 抓取物体 | 9,381 |
| 4 | 机械臂运动时未滑动 | 61,365 |
| 5 | 机械臂运动时滑动 | 34,835 |
| 6 | 其他触觉事件，二分类中排除 | 943 |

## 导出格式

`processed/<物体>_Pose<姿态>_Exp<试验>.npz` 每个文件对应一次完整试验：

- `x`：`(T,54)`，float32；通道按触点展开，每点依次为 X、Y、Z。已逐值验证从 float64 转为 float32 没有损失。
- `label`：`(T,)`，int8，原标签 0–6。
- `timestamp_s`：`(T,)`，float64，原时间戳转为秒。
- `slip`：`(T,)`，int8，标签属于 1 或 5 时为 1。
- `valid`：`(T,)`，bool，仅标签 6 为无效。
- `onset`：`(T,)`，bool，当前帧开始滑动且上一帧为有效非滑动。记录开始时已经滑动的帧不计作起点，共 238 个起点。

未进行归一化、滤波、重采样或学习特征。训练期的缩放参数必须仅由训练集计算；未来目标只能读取未来标签，输入窗口不能含有未来信号。完整逐试验统计在 `arq2021_audit.json` 和 `trials.csv`。

`splits.json` 提供在模型训练前固定的两种划分：

- 按试验留出：每个物体和姿态的 Exp1–6 训练、Exp7–8 验证、Exp9–10 测试，分别 54/18/18 次试验；滑动起点数 154/45/39。
- 留一物体：每次用一个完整物体作测试；另外两个物体 Exp1–8 训练、Exp9–10 验证。

窗口不允许跨试验或跨集合。帧数不能当作独立样本数；置信区间至少应以试验为重采样单位。三个留物体结果也不足以推断广泛的物体总体。

26/90 次试验短于 5 秒，且不同物体的短试验比例不同。比较历史长度时应使用相同、具备足够历史的目标帧，或明确报告起始掩码及覆盖率；不能用未说明的零填充制造长历史收益。该观察数据不能自行识别“真正有效记忆长度”，也不能独立调节物理事件稀疏度。

## FlexiTac 的适用边界

已实时核对 [Tna001/tactile_test_tube_pyflexitac](https://huggingface.co/datasets/Tna001/tactile_test_tube_pyflexitac)，当前提交 `ebd3c711d678aa18bfc7f6a0a1e894104c8c1ea4` 与既有缓存一致，数据卡标注 Apache-2.0。仅只读检查旧项目的 10 个 parquet、元数据和数组，没有修改旧项目。

实际字段含触觉 12×32、动作、关节状态、时间与序号；共有 64 回合、75,041 帧、30 Hz。发布模式和实际 parquet 中均未发现独立的接触、滑动、失败或成功标签。详情及缓存文件哈希见 `flexitac_label_audit.json`。

因此，这一现成版本不能直接作为独立接触起点/滑动/成功实验的验证集。对触觉自身阈值化生成标签只能验证该规则的恢复；预测未来关节动作会改变研究问题。其视频可能支持另行人工标注，但本次未做此工作。30 Hz 的记录每 100 ms 约 3 帧，也不能由元数据推断原生传感器的端到端延迟。

## 复现

```powershell
python -m pip install --target work/real_data/deps --no-deps h5py
python work/real_data/acquire.py --download
python work/real_data/preprocess.py
```

还需要 numpy、requests、pyarrow；本机系统 Python 已具备。`preprocess.py` 中 FlexiTac 审计使用已存在的只读缓存绝对路径，异机复现需要修改该缓存路径，或只运行 ARQ 数据导出。
