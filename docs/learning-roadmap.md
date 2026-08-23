# Robot Learning → Physical Intelligence 学习与复现路线

> 更新日期：2026-08-23  
> 目的：为后续机器人学习、VLA、World Model、工具使用泛化与 Physical Intelligence 研究建立可执行的学习路线。

---

## 0. 学习方法：不要只“读论文”，也不要只“跑代码”

推荐采用下面的循环：

```text
学习基础概念
    ↓
阅读一篇代表论文
    ↓
复现最小实验
    ↓
主动修改一个变量
    ↓
观察性能和失败模式
    ↓
尝试解释原因
    ↓
形成新的研究问题
```

把“复现”分成三层：

| 层级 | 示例 | 价值 |
|---|---|---|
| 工程复现 | clone 仓库，成功训练 ACT | 熟悉工程流程 |
| 机制复现 | 比较 chunk size = 1/16/64 | 理解方法为什么有效 |
| 研究复现 | 改训练分布，验证论文解释是否仍成立 | 开始形成研究能力 |

长期应该逐渐从第一层进入第三层。

---

# 第一阶段：Imitation Learning 与 Robot Policy

核心问题：

> **机器人怎样从示教数据中学习动作？为什么普通监督学习在机器人上容易失败？**

## 1. Behavior Cloning 与 Compounding Error

### 推荐阅读

**DAgger**  
Ross et al., 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*  
https://proceedings.mlr.press/v15/ross11a.html

### 重点理解

普通 Behavior Cloning 学：

```text
observation → action
```

但训练状态分布和机器人真正部署后的状态分布不同：

```text
p_train(s) ≠ p_policy(s)
```

一次小误差可能把机器人带进训练数据没有覆盖的状态，随后误差继续放大，即：

- covariate shift
- compounding error

### 不建议

不需要一开始啃完整理论证明，也不必正式复现 DAgger。

### 建议的小实验

自己写一个二维导航 / 小车任务：

1. 专家产生一批理想轨迹；
2. 用简单 MLP 做 BC；
3. 测试时改变初始位置一点点；
4. 观察误差是否逐步累积。

目标：亲眼看到“离线监督损失很低 ≠ rollout 一定成功”。

---

## 2. ACT：第一个正式复现项目

### 论文

Zhao et al., 2023, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*  
ACT 官方仓库：  
https://github.com/tonyzhaozh/act

### 核心思想

不要只预测下一步动作：

```text
o_t → a_t
```

而是一次预测一段动作：

```text
o_t → (a_t, a_{t+1}, ..., a_{t+H})
```

也就是 Action Chunking。

### 建议复现方式

先跑官方 MuJoCo 仿真：

- Transfer Cube
- Bimanual Insertion

完整经历：

```text
生成/读取 demonstration
→ dataset
→ train
→ checkpoint
→ rollout
→ success rate
```

### 跑通后必须做的消融

固定任务和数据，改变：

```text
chunk size = 1 / 4 / 16 / 32 / 64
```

再逐渐改变：

- demonstration 数量；
- observation history；
- 初始位置扰动；
- action noise；
- temporal ensembling 是否启用。

不要只记录 success rate，也记录：

- 动作是否抖动；
- 哪个阶段最容易失败；
- 失败后能否恢复；
- validation loss 是否真的能预测 rollout 成功率。

---

## 3. Diffusion Model 基础：先做 Toy Example

### 推荐基础论文

Ho et al., 2020, *Denoising Diffusion Probabilistic Models*

### 学习要求

不用一开始研究图像大模型。

先对二维数据实现最小 diffusion：

- 两个高斯簇；
- two moons；
- 简单二维轨迹分布。

理解：

```text
clean data
→ gradually add noise
→ learn denoising
→ Gaussian noise
→ sample data
```

目标不是生成好看的图片，而是搞懂：

- forward process；
- reverse process；
- noise prediction；
- conditional generation；
- multimodal distribution。

---

## 4. Diffusion Policy：第二个正式复现项目

### 论文

Chi et al., 2023, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*  
https://arxiv.org/html/2303.04137

### 为什么重要

机器人同一状态下可能存在多个正确动作。

例如绕障碍物：

```text
左边绕：正确
右边绕：正确
```

普通 MSE regression 可能学习二者平均值，而平均动作本身可能是错误的。

Diffusion Policy 学习完整条件动作分布：

```text
p(action chunk | observation)
```

### 重点概念

- multimodal action distribution
- conditional diffusion
- action chunk
- observation horizon
- prediction horizon
- execution horizon
- receding horizon control
- closed-loop replanning

### 推荐实验

尽量在**与 ACT 相同的任务和数据**上测试：

```text
BC vs ACT vs Diffusion Policy
```

不要为了“跑更多模型”而不停换 benchmark。

只有任务、数据、评估方式尽量一致，才有机会理解算法本身的差异。

### 特别建议设计一个双模态任务

例如机器人需要从障碍物左/右两侧到达目标。

观察：

- BC 是否平均；
- ACT 怎样处理；
- Diffusion Policy 是否会稳定采样一种合理模式。

---

## 5. Flow Matching：为 π₀ 和连续动作模型做准备

### 论文

Lipman et al., 2023, *Flow Matching for Generative Modeling*  
https://doi.org/10.48550/arxiv.2210.02747

### 核心直觉

Diffusion 可以理解为逐步把噪声变成数据。

Flow Matching 则直接学习一个时间相关向量场：

```text
v_theta(x_t, t)
```

告诉样本：

> 当前在这里，下一瞬间应该向哪个方向“流”。

然后通过 ODE 从简单分布运输到数据分布。

### 推荐动手

继续沿用前面的二维数据：

```text
Toy DDPM
vs
Toy Flow Matching
```

画出：

- 数据点；
- 速度场；
- 从 noise 到 data 的轨迹；
- sampling steps 与误差/速度的关系。

做到这一步，再读 π₀ 会容易很多。

---

# 第二阶段：Generalist Robot Policy 与 VLA

核心问题：

> **当机器人数据从一个任务扩展到很多任务、很多环境、很多机器人后，模型为什么可能得到更强的泛化？**

---

## 6. Open X-Embodiment：重点学习“数据”

### 项目

*Open X-Embodiment: Robotic Learning Datasets and RT-X Models*  
https://robotics-transformer-x.github.io/

### 阅读重点

不要只看 RT-X 模型结构。

更重要的是理解：

- 不同机器人 embodiment 的数据如何统一；
- action space 怎么标准化；
- 相机数量不同怎么办；
- proprioception 不同怎么办；
- 为什么 cross-embodiment co-training 可能产生 positive transfer；
- 数据量和数据 diversity 哪个更重要；
- 数据重复度与 data mixture 权重有什么影响。

### 与未来研究的连接

这是研究以下问题的基础：

```text
Data Scale
Data Diversity
Task Diversity
Embodiment Diversity
↓
Generalization
```

---

## 7. Octo：最值得亲手 finetune 的 Generalist Policy

### 论文 / 项目

*Octo: An Open-Source Generalist Robot Policy*  
https://octo-models.github.io/  
https://github.com/octo-models/octo

### 核心组成

```text
Transformer backbone
+
Diffusion action head
+
Open X-Embodiment pretraining
```

训练数据约 800k trajectories。

### 为什么特别适合学习

它提供：

- Octo-Small（27M）
- Octo-Base（93M）
- pretrained checkpoint
- inference notebook
- finetune 示例
- Open X data loader

### 正确的动手方式

不要完整重训 800k 数据。

做：

```text
pretrained Octo
→ small downstream dataset
→ finetune
```

然后和：

```text
same architecture from scratch
```

比较。

### 建议实验

比较：

- 10 demos
- 25 demos
- 50 demos
- 100 demos

看看 pretrained model 与 scratch model 的 sample efficiency 差距。

第一次亲手验证：

> 大规模预训练到底有没有带来 transferable knowledge？

---

## 8. RT-2：重点理解 Semantic Transfer

### 论文

*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*

### 阅读问题

RT-2 的重点不是复制模型，而是理解：

```text
Internet-scale VLM knowledge
+
robot trajectory data
→
robot actions
```

特别注意区分：

```text
Semantic Generalization
```

和：

```text
Physical Skill Generalization
```

模型知道“石头可以临时当锤子”，不代表它真正知道怎样产生合适的接触、力和运动。

这是未来工具使用研究的重要分界线。

---

## 9. OpenVLA：理解 VLM 如何变成 VLA

### 论文 / 项目

*OpenVLA: An Open-Source Vision-Language-Action Model*  
https://openvla.github.io/

### 核心结构

```text
RGB
→ SigLIP + DINOv2
→ Projector
→ Llama 2
→ tokenized actions
→ continuous robot actions
```

训练约 970k robot trajectories。

### 推荐做法

- 阅读完整架构；
- 跑 inference；
- 后续条件允许时 LoRA finetune；
- 不尝试完整复现预训练。

完整训练需要大规模多 GPU 算力，现阶段价值不高。

---

## 10. π₀：理解现代连续 VLA

### 论文

*π₀: A Vision-Language-Action Flow Model for General Robot Control*  
https://arxiv.org/html/2410.24164v3

### 阅读前置

最好已经懂：

- Transformer
- VLM / VLA 基础
- action chunking
- diffusion
- flow matching

### 核心思想

```text
pretrained VLM
+
flow-matching action expert
+
large-scale multi-robot data
```

用于连续、高频的机器人动作生成。

### 推荐

读懂，不做完整预训练复现。

重点研究：

- 为什么不直接 action token；
- 为什么使用 flow matching；
- pretraining / post-training 的角色分别是什么；
- cross-embodiment data 怎样进入同一个模型。

---

## 11. 方法论论文：What Matters in Building VLA

### 论文

*What Matters in Building Vision-Language-Action Models for Generalist Robots*  
https://arxiv.org/html/2412.14058

### 为什么值得认真读

论文进行了大量 controlled experiments，对：

- VLM backbone
- VLA architecture
- history modeling
- continuous / discrete action
- cross-embodiment data
- pretrain / finetune / post-train

做系统比较。

对于未来研究“物理智能为什么出现”，它最大的价值不是某一个最终数字，而是学习：

> 如何通过 ablation 把多个可能的因果因素拆开。

---

# 第三阶段：World Model 与动作后果预测

核心问题：

> **机器人能否不仅学“看到这个就做这个动作”，还学会“如果我这样做，世界会发生什么”？**

---

## 12. 先读 World Model Survey 建立地图

### 推荐

*World Models for Robotic Manipulation: A Survey*  
https://arxiv.org/html/2606.00113

### 阅读时始终问三个问题

#### 1. 它预测什么？

- pixels / video
- latent state
- 3D / point cloud
- 4D scene
- structured physical state

#### 2. action 怎样进入模型？

```text
(current world, action)
→ predicted future world
```

#### 3. 预测结果怎么帮助机器人？

- policy representation
- planning
- candidate action evaluation
- synthetic data generation
- policy evaluation
- model-based RL

---

## 13. IRASim：第一个重点 World Model 方法

### 论文

*IRASim: A Fine-Grained World Model for Robot Manipulation*  
https://arxiv.org/html/2406.14540

### 核心问题

给定：

```text
historical observation + future action trajectory
```

预测：

```text
future video
```

也就是：

```text
(o_history, a_future)
→ future world
```

### 不建议一开始复现完整视频生成模型

先自己做一个简化 dynamics model。

例如仿真中：

```text
robot + stick + cube
```

state：

```text
cube pose
stick pose
contact state
```

action：

```text
Δx, Δy, Δtheta
```

预测：

```text
next state
```

然后改变：

- 工具长度；
- 工具宽度；
- 摩擦系数；
- 接触点；
- 物体质量。

测试 dynamics model 对未见组合的预测误差。

这已经是一个非常好的“微型 Physical Intelligence”实验。

---

## 14. DreamZero / World Action Model

### 论文

*World Action Models are Zero-shot Policies*  
https://arxiv.org/html/2602.15922

### 核心视角

普通 VLA 更擅长：

```text
semantic generalization
```

World Action Model 希望同时学习：

```text
future world + robot action
```

让动作和物理世界演化被共同建模。

### 阅读时重点观察

- video pretraining 到底带来了什么；
- “物理先验”究竟如何被证明；
- 新任务泛化是否来自 motion knowledge，而不仅仅是语义知识；
- 对训练数据 diversity 的依赖。

---

# 第四阶段：Affordance 与 Tool Generalization

核心问题：

> **机器人如何不再记住“这是某个工具”，而是学习“这个物体的什么性质能实现什么物理效果”？**

这是长期最值得持续追踪的一条主线。

---

## 15. AffordDP：把 Affordance 接到 Diffusion Policy 上

### 论文

*AffordDP: Generalizable Diffusion Policy with Transferable Affordance*  
https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_AffordDP_Generalizable_Diffusion_Policy_with_Transferable_Affordance_CVPR_2025_paper.pdf

### 关键表示

不仅预测动作，还显式引入：

```text
3D contact point
+
post-contact trajectory
```

即：

```text
Where to interact?
How to interact?
```

### 重点思考

为什么显式 affordance representation 会比：

```text
RGB → Action
```

更容易泛化？

---

## 16. FUNCTO / MimicFunc：Functional Correspondence

### 论文

*FUNCTO: Function-Centric One-Shot Imitation Learning for Tool Manipulation*  
https://arxiv.org/html/2502.11744

*MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence*  
https://proceedings.mlr.press/v305/tang25a.html

### 核心问题

两个工具可能外观差异很大，但功能相同。

例如：

```text
mug
teapot
```

完成“倒”这一功能时，都有：

- grasp region
- functional region
- functional orientation
- tool-target relation

模型真正应该匹配的是：

```text
functional correspondence
```

而不是 visual similarity。

---

## 17. FORGE：长期重点论文

### 论文

*FORGE: Towards Functional Tool-Use Generalization via Keypoint Trajectory Reasoning*  
https://arxiv.org/html/2607.05780

### 关键概念

```text
Functional Generalization
```

不同工具长得不同，但可以实现同一个功能。

FORGE 比较了不同中间表示：

- affordance image
- human video prompt
- 2D keypoint trajectory

其核心路线是：

```text
Visual Observation
→ Functional Keypoint Trajectory
→ Robot Action
```

而不是直接：

```text
Visual Observation
→ Robot Action
```

### 长期研究启发

可能真正值得研究的问题是：

> 什么样的 intermediate representation 能迫使/引导网络学习可迁移的物理功能，而不是记忆工具类别和训练轨迹？

---

## 18. GROW²：Which Tool? Where to Use It?

### 论文

*GROW²: Grounding Which and Where for Robot Tool Use*  
https://arxiv.org/html/2606.30632v1

### 两个问题

```text
Which?
```

从开放场景里选择哪个物体作为工具。

```text
Where?
```

使用该物体的哪个部位完成任务。

### 重要思想

把 Object Part 当作 intermediate abstraction：

- handle
- blade
- tip
- rim

这再次说明：

> 适当的中间表示可能比单纯增大 end-to-end 模型更重要。

---

# 第五阶段：自己真正值得做的受控实验

## 实验 A：Data Quantity vs. Data Diversity

这是目前最推荐的第一个自主研究型小实验。

### 核心问题

> 在总数据量相同时，更多工具种类是否比更多重复轨迹更能促进未见工具泛化？

### 环境

推荐使用 ManiSkill，建立一个简化工具任务：

```text
Robot + Tool → Push Object To Goal
```

固定：

- robot embodiment
- policy architecture
- task definition
- camera
- training steps
- total trajectory count

只改变工具 diversity。

### 示例实验组

保持总轨迹数 10,000：

```text
Group A: 5 tools × 2000 trajectories
Group B: 20 tools × 500 trajectories
Group C: 100 tools × 100 trajectories
```

测试只使用完全未见工具。

观察：

```text
Tool Diversity
→ Unseen Tool Success Rate
```

### 第二个轴：Data Quantity

固定工具 diversity，例如 20 tools：

```text
1k / 5k / 10k / 50k trajectories
```

这样才能把：

```text
Quantity
```

和：

```text
Diversity
```

拆开。

---

## 实验 B：Direct Action vs. Functional Representation

比较：

```text
RGB → Action
```

和：

```text
RGB
→ Functional Representation
→ Action
```

Functional Representation 可以逐渐尝试：

1. tool ID；
2. continuous geometry vector；
3. contact point；
4. functional keypoints；
5. keypoint trajectory；
6. contact + force / motion effect。

测试：

```text
seen tool
unseen instance
unseen category
```

的性能变化。

---

## 实验 C：World Model 是否帮助 Tool Generalization

训练：

```text
(current state, action, tool geometry)
→ next state
```

然后测试：

- 未见长度；
- 未见形状；
- 未见质量；
- 未见摩擦；
- 未见组合。

再比较：

```text
Policy only
vs
Policy + World Model
```

目标是开始回答：

> 对动作结果的显式预测，是否比单纯 BC 更有利于物理泛化？

---

# 推荐工具栈

| 工具 | 推荐程度 | 当前主要用途 |
|---|---:|---|
| LeRobot | ★★★★★ | ACT / Diffusion / 数据采集入门 |
| ManiSkill | ★★★★★ | 自主受控实验、大规模仿真数据 |
| robomimic | ★★★★☆ | 标准 imitation learning benchmark |
| LIBERO | ★★★★☆ | 多任务与 generalization |
| Octo | ★★★★☆ | generalist policy finetune |
| RoboTwin 2.0 | ★★★☆☆ | 后期复杂任务 / 双臂 / cross-embodiment |
| OpenVLA / π₀ | ★★★☆☆ | inference / finetune / 架构学习 |

### 主要项目

LeRobot：  
https://github.com/huggingface/lerobot

ManiSkill：  
https://github.com/haosulab/ManiSkill

robomimic：  
https://github.com/ARISE-Initiative/robomimic

LIBERO：  
https://github.com/Lifelong-Robot-Learning/LIBERO

Octo：  
https://github.com/octo-models/octo

RoboTwin：  
https://github.com/RoboTwin-Platform/RoboTwin

---

# 最值得优先阅读的 10 篇 / 项目

| 顺序 | 论文 / 项目 | 学习方式 |
|---:|---|---|
| 1 | DAgger | 理解问题，不重点复现 |
| 2 | ACT | **完整复现 + 消融** |
| 3 | Diffusion Policy | **完整阅读 + 复现** |
| 4 | Flow Matching | 数学理解 + toy implementation |
| 5 | Open X-Embodiment | 重点研究数据与 generalization |
| 6 | Octo | **finetune + 小规模 ablation** |
| 7 | π₀ | 理解架构与训练逻辑 |
| 8 | World Models for Robotic Manipulation Survey | 建立世界模型知识地图 |
| 9 | AffordDP | 从 policy 进入 affordance |
| 10 | FORGE | 长期工具泛化研究重点 |

扩展阅读：

- RT-2
- OpenVLA
- IRASim
- MimicFunc / FUNCTO
- GROW²
- DreamZero

---

# 论文阅读固定提问模板

以后每篇 Robot Learning 论文至少回答：

## 1. 它解决了什么以前解决不了的问题？

不要写“提出了一个新模型”。

要写真正的 failure mode。

## 2. 作者认为 bottleneck 是什么？

例如：

- data scarcity
- covariate shift
- multimodal action
- weak visual representation
- lack of dynamics understanding
- poor affordance representation

## 3. 作者加入了什么 inductive bias / representation？

例如：

- action chunk
- diffusion distribution
- flow vector field
- keypoint
- contact point
- object part
- future video
- latent dynamics

## 4. 哪个 ablation 真正支撑了作者的解释？

不是看最好结果，而是看：

> 去掉某一设计以后发生什么？

## 5. 如果改变训练数据分布，结论还成立吗？

这是未来特别值得长期追问的问题。

---

# 长期研究视角

希望逐步把思考从：

```text
Which architecture gets the best success rate?
```

转向：

```text
What data creates what representation?

What representation enables what kind of generalization?

Under what conditions does physical intelligence emerge?
```

最终关注的关系可以概括为：

```text
Data
↓
Representation
↓
Generalization
↓
Functional / Physical Intelligence
```

以及工具使用中的：

```text
Tool Properties
+
Contact
+
Action
↓
Physical Effect
↓
Task Success
```

目标不是让机器人记住“刷子用来扫、锤子用来敲”，而是逐渐研究它能否学习：

> **什么物理结构，通过什么接触和运动，会产生什么效果。**
