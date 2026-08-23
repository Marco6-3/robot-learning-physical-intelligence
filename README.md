# Robot Learning & Physical Intelligence Notes

> 一个用于长期记录机器人学习、VLA、World Model、工具使用泛化与 Physical Intelligence 相关学习、论文阅读、代码复现和实验心得的私人研究仓库。

## 仓库目标

这个仓库不是单纯的“论文收藏夹”，而是希望形成一条从基础学习到自主研究的连续记录：

1. 学习机器人策略如何从示教数据中学习动作；
2. 理解大规模、多任务、多机器人数据如何形成可迁移能力；
3. 学习 World Model 如何预测“动作会让世界发生什么变化”；
4. 进一步研究工具使用、affordance 与 functional generalization；
5. 最终尝试回答一个更长期的问题：**大规模具身交互数据在什么条件下会促使机器人形成可泛化的物理智能？**

当前重点关注的两个研究问题：

- **Physical Intelligence Emergence**：数据规模、数据多样性、模型结构与训练目标分别怎样影响机器人物理泛化能力？
- **Generalizable Tool Use**：机器人怎样学习“工具—动作—效果”的关系，并把这种关系迁移到未见工具与未见任务？

## 当前进度摘要

截至 **2026-08-23**：

- [x] 已明确长期研究兴趣：机器人学习、物理智能涌现、工具使用与功能泛化；
- [x] 已整理从 Transformer 之后进入 Robot Learning 的整体学习路径；
- [x] 已使用 Exa 筛选第一批核心论文与开源项目；
- [x] 已确定学习方式：**基础概念 + 论文阅读 + 最小复现 + 主动消融实验**，而不是只跑通作者代码；
- [ ] 系统完成 Transformer 学习；
- [ ] 完成第一个 ACT 仿真复现；
- [ ] 在同一任务上比较 BC / ACT / Diffusion Policy；
- [ ] 实现一个二维 toy diffusion / flow matching 实验；
- [ ] 在 ManiSkill 中搭建简化工具使用任务；
- [ ] 开始第一个受控 scaling experiment：Data Quantity vs. Data Diversity；
- [ ] 进入 Octo / OpenVLA 等 generalist policy 的小规模 finetune；
- [ ] 开始 World Model 与 Tool Generalization 专题实验。

## 学习主线

```text
Transformer
    ↓
Behavior Cloning / Imitation Learning
    ↓
ACT / Action Chunking
    ↓
Diffusion Policy
    ↓
Flow Matching
    ↓
Generalist Robot Policy / VLA
    ↓
World Model / World Action Model
    ↓
Affordance / Functional Representation
    ↓
Generalizable Tool Use
    ↓
Physical Intelligence
```

详细论文顺序、复现方式和实验建议见：

- [`docs/learning-roadmap.md`](docs/learning-roadmap.md)

## 推荐的记录方式

以后每完成一篇论文或一个实验，不只记录“跑通了没有”，至少写清楚：

- 论文/实验解决了什么问题；
- 核心假设是什么；
- 输入、输出、action representation 是什么；
- 作者真正加入了什么 inductive bias；
- 哪个 ablation 最能支撑论文结论；
- 自己复现时改了什么变量；
- 结果与预期是否一致；
- 失败原因；
- 对自己的研究问题有什么启发。

实验笔记模板见：

- [`notes/_template.md`](notes/_template.md)

## 近期第一阶段目标

第一阶段不追求训练大模型，目标是先真正理解机器人策略学习：

1. 用 LeRobot / MuJoCo 跑通 ACT；
2. 改 action chunk size、训练数据量、初始状态扰动；
3. 在相同任务上复现 Diffusion Policy；
4. 比较普通 BC、ACT、Diffusion Policy 的失败模式；
5. 学会从 rollout 结果反推数据分布、action representation 和 closed-loop execution 的问题。

第一阶段完成后，再进入 Octo、OpenVLA、π₀、World Model 与工具泛化。

## 仓库结构

```text
.
├── README.md
├── docs/
│   └── learning-roadmap.md
└── notes/
    └── _template.md
```

后续可以逐渐扩展成：

```text
papers/          # 单篇论文阅读笔记
reproductions/   # 复现实验记录
experiments/     # 自主实验
figures/         # 实验图片与图表
configs/         # 关键训练配置
scripts/         # 自己编写的实验脚本
ideas/           # 研究问题与假设
```

## 原则

> **不要把“跑通代码”当成复现的终点。**

更重要的是：固定大部分条件，主动改变一个变量，然后解释结果为什么变化。

长期希望把研究视角从：

```text
Architecture → Success Rate
```

逐步转向：

```text
Data → Representation → Generalization → Emergent Physical Intelligence
```
