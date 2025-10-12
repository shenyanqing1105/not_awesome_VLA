# not_awesome_VLA
record papers and ideas, focusing on vision-language-action (VLA) models

## List
skill memory

## Summary
RDT系列

- RDT-1B

  
- RDT2

- **硬件革新**：对UMI（Unified Manipulation Interface）硬件进行了重新设计
- **海量多样数据：**在100+室内场景中收集了超1万小时的人类操作视频，覆盖了抓取器能完成的绝大多数家用任务，数据采集成本仅为传统遥操作的1/10，速度提升5倍。
- 三阶段训练
  - **VLA预训练** RDT2-VQ：在Qwen2.5-VL-7B-Instruct模型上预训练了UMI数据集，以图像和语言作为输入，输出离散的动作标记（残差向量量化（RVQ））
  - **扩散：RDT2-FM**用一个**400M参数的RDT模型**（RDT-1B的改进版）替换RVQ，作为动作专家，attend到主干KV；冻结预训练的VLM主干，采用流匹配损失函数
  - **蒸馏RDT2-UltraFast**：将RDT2-FM蒸馏为**一步扩散策略**，生成**RDT2-UltraFas**t模型，保持性能无损。Qwen主干依然冻结，模型能将纯噪声直接映射为机器人动作，仅需**单次扩散步骤**，类似于GAN的生成机制。

实验结果

- 相变点**（**Phase Transition Point），展示了模型从训练中直接展现的鲁棒零样本泛化能力；【未知形态、未知场景、未知物体、未知语言】
- 能力边界（Capability Boundary），通过六项挑战性下游任务评估RDT2-UltraFast的在分布内性能。

  
- MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation [[paper]](http://arxiv.org/abs/2508.19236)
> introducing working memory and memory bank for long-horizon tasks [working memory includes perceptual tokens and cognitive tokens; bank corresponds to hippocampus, storing precise details and abstract semantics]
> 
> manipulation属于非马尔科夫过程，指当前状态的转移概率不仅依赖于当前状态，还依赖于之前的状态，具有记忆效应的随机过程；这里的记忆指的是同一episode内的记忆，训练时采用streaming queue作为dataloader;
>
> retrieval是working memory（Q，dual）和bank中的两类token（KV）做的cross-attention的过程（并非单纯的“检索”）；gate fusion是通过learned gate把working memory和retrieved embeddings结合到一起，得到memory-augmented representations；memory consolation是当bank内memory长度超过L时做特征合并，将相邻两帧的memory-augmented representations做相似度比较，最高的一对将被合并为它们的均值。
<img width="500" alt="image" src="https://github.com/user-attachments/assets/1e03f0ca-8b95-4996-be8d-62893ab0549d" />

#### 世界模型相关
- uniVLA

#### 推理链相关chain-of-thought
cot-vla、embodied cot、tracevla【手工设计中间表示】
- Chain-of-Action
> compounding error复合误差
> 
> 动作建模：逆向地从“关键帧”动作开始，自回归生成动作轨迹；latent consistency；自回归如何连续动作？
> 
> reverse的优势：goal-conditioned（action、image），spatial continuity
>
> 不足：训练过程中关键帧需要人工选择标记【decode出的第一个token就是下一个关键帧】
> 
> 未来：任务意图的嵌入构建，建立上层规划到动作生成的约束

#### 其他

- FiS-VLA, Fast-in-Slow
> 把S1和S2的工作流统一在一个VLM中，即VLM完成了大脑（浅脑、深脑）：将S1的快速执行能力整合进一个预训练的VLM中，同时保留本身的S2推理能力
> 具体来说，FiS-VLA将S2的最后几层的Transformer模块重新利用，并构建为一个高效执行的S1；充分利用VLM的推理能力，同时避免引入新的、未经过大规模预训练的S1
> S1的输入依然是多模态的encoder结果
> 双系统感知协同训练策略：扩散建模损失监督动作生成+自回归next-token预测生成离散动作或语言
<img width="1002" height="423" alt="image" src="https://github.com/user-attachments/assets/2d045802-d782-46ee-aed1-47b70754b598" />

## 积累
关于机器人的研究，已经从“本体的运动机能”转向了“智能体的感知决策”。

五大元素：数据、输入、模型架构、输出、训练策略

#### 具身智能的五级分类体系
   
- L1（单一任务完成）：机器人能够可靠地完成单一、明确定义的任务（如抓取物体），但功能局限于特定任务领域。
- L2（组合任务完成）：机器人能够处理组合任务，通过将高级人类指令分解为简单动作序列来执行，但能力仍限于预定义任务和技能库。
- L3（有条件的一般任务完成）：机器人能够处理多种任务类别，表现出对任务、环境和人类指令的有条件泛化能力，但对全新或开放式任务的可靠性不足。
- L4（高度通用机器人）：机器人展现出对广泛未见任务的稳健泛化能力，具备强大的多模态理解和推理能力。
- L5（全功能机器人）：这是具身AGI的最终目标，机器人能够满足人类日常生活的广泛需求，展现出类似人类的认知行为。

#### 技术路线：分层决策 or 端到端？
   
- 分层决策：将任务分解为多个层次，比如有一个思考拆解任务的大脑+执行具体任务并生成动作的小脑+whole-body controller；不同步骤间的对齐和一致性需解决
- 端到端：从VL、state输入直接映射到最终的动作序列，无需显式的任务分解；存在黑箱效应
   
#### 训练方法：模仿学习 or 强化学习？
   
- IL：通过观察专家演示行为来学习执行任务的方法。可以快速学习专家策略，无需复杂的探索过程，适用性强，适合试错代价高的任务；需要大量高质量的专家演示数据，对于未见过的情况泛化能力较差
- RL：强化学习是一种通过智能体与环境交互来学习最优策略的方法。通过探索环境学习未知的策略，可以处理高度不确定和动态变化的环境，具备较强的泛化能力；为复杂任务设计合适的奖励函数难度较高，学习效率低，训练过程不易保证稳定性
   
#### 数据
   
   自主数据闭环：通过商业化落地，在真实使用场景中收集数据、优化模型，形成数据飞轮。落地形式有通用场景软硬结合、纯软件路径、垂直领域软硬结合等。

   采集：本质是对 “成本、精度、泛化能力/规模” 三者的平衡。
   | 操作方式 |  实现形式  |  优势| 目前不足    |
   | ---- | ---- | ---- | ---- |
   | 遥操作 |  VR 遥操 + 动捕手套 | 高精度 | 高成本 |
   | 仿真     |  构建虚拟环境生成数据    |  低成本|无法完全模拟真实世界的物理规律    |
   |  UMI 多模态传感器融合    | “视觉 - 惯性融合”的末端执行器跟踪     |  全身动作捕捉能力不足    |
   |  视频学习   | 低成本、效率高     |  数据标注难、视频处理要求高    |

   如何learn from human video（third-view）？视觉信息如何转换？

   数据规模？单条数据精度？
   
#### 落地场景
   - 工业应用-场景比较固定、可控
   - 任务相对清晰的生活场景-咖啡、饮料制作、冰激凌等
   - 清洁机器人
   - 配送机器人

#### 发展的主流框架及局限及改进的关键问题

**传统LVA**，如openVLA，没有双系统的架构区分，action当做token来预测；大模型前向传播耗时长、对VLM的丰富预训练知识的利用不足

**基于人脑思维研究的快慢系统·分层架构**

- 系统1，快执行，根据模式和经验快速做出决策和判断，低级动作生成；快速执行、原子规划、精确操作；潜意识、直觉、运动控制、快速响应
- 系统2，慢思考，分析、拆解复杂任务，高级任务推理；场景理解、长程规划、任务拆分；有意识、刻意思考深度

关键科学问题：如何充分利用S2的推理能力，如何保证S1的精准执行能力，如何构建双系统之间的连接

- 同步双系统架构，VLM+action expert，如π0、GR00T N1；统一从S2输入视觉观测，S2到S1是多模态特征；S1 S2的频率是一致的，整体串联
- 同步子任务分解双系统架构，如π0.5；S2负责长程复杂任务/抽象任务到原子任务的拆解与特征提取，S1负责原子任务的准确执行；通过原子任务拆分以及latent feature连接双系统，输入依然是同频
- 异步双系统架构，如Helix、HiRT；S1能获取完整的、高频的多模态输入（轻量encoder），同时接收S2的latent feature（一步输出、多步复用）；通过特征连接双系统

## 想法记录
如何结合自身过往经验？运动趋势/模式/方向如何表征？

小孩如何学习的技能？如何在网络层面做到举一反三？

大脑的运作形式？大小脑、快慢系统？如何连接快慢脑，实现慢脑规划和快脑操作的模态对齐？

如何突破模仿学习的瓶颈，训练丰富技能？RL？
