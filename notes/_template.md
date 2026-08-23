# 实验 / 论文复现笔记模板

> 日期：YYYY-MM-DD  
> 类型：论文阅读 / 工程复现 / 机制复现 / 自主实验

## 1. 基本信息

- 论文 / 项目：
- 链接：
- 代码仓库：
- 当前代码 commit：
- 环境：
- GPU：
- 任务：

## 2. 这项工作真正解决什么问题？

用自己的话描述 failure mode，不要直接抄摘要。

## 3. 核心假设

作者认为性能瓶颈是什么？

## 4. 方法

### Input

### Output

### Observation Representation

### Action Representation

### Network / Policy

### Loss

### Training Data

## 5. 最重要的 Inductive Bias / Intermediate Representation

例如：

- action chunk
- diffusion
- flow matching
- affordance
- keypoint
- contact
- world prediction

## 6. 复现设置

### Dataset

### Train / Val / Test Split

### Hyperparameters

### Evaluation Metric

### Random Seeds

## 7. 复现结果

| Setting | Metric | Result |
|---|---|---|
| Baseline | | |
| Reproduction | | |

## 8. 与论文结果的差异

- 一致的部分：
- 不一致的部分：
- 可能原因：

## 9. 主动消融实验

### Hypothesis

> 如果改变 ______，我预计 ______，因为 ______。

### Controlled Variables

保持不变：

- 
- 

只改变：

- 

### Results

| Variable | Result | Failure Mode |
|---|---:|---|
| | | |

## 10. Failure Analysis

不要只记录“失败”。

分类：

- perception error
- wrong target
- trajectory error
- compounding error
- contact failure
- gripper failure
- OOD state
- action jitter
- timing / control issue
- model hallucination
- other

## 11. 我学到了什么？

### 关于算法

### 关于数据

### 关于机器人系统

### 关于泛化

## 12. 对 Physical Intelligence / Tool Use 的启发

这篇论文或这次实验是否说明：

- 数据量重要？
- 数据 diversity 重要？
- intermediate representation 重要？
- world prediction 重要？
- explicit physical structure 重要？
- end-to-end 模型出现了 shortcut learning？

## 13. 下一步问题

1. 
2. 
3. 
