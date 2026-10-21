<!-- Translation of README.md. Update it in the same change as README.md. -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" alt="Syntropy" width="360">
  </picture>
</p>

<p align="center"><a href="README.md">English</a>&emsp;<b>简体中文</b>&emsp;<a href="README.ja.md">日本語</a>&emsp;<a href="README.de.md">Deutsch</a>&emsp;<a href="README.fr.md">Français</a>&emsp;<a href="README.es.md">Español</a>&emsp;<a href="README.pt-BR.md">Português</a></p>

<p align="center"><sub>本文是译文，内容以<a href="README.md">英文 README</a> 为准。</sub></p>

<h3 align="center">用你拥有的每一块 GPU、每一个存储层级和每一台机器，运行同一个大模型</h3>

<p align="center">放置方案由实测你的硬件决定。默认输出与参考结果一致。<br>
以 Rust 编写，为 CUDA、HIP、Metal、Vulkan 和 CPU 提供原生内核。</p>

<p align="center">
  <a href="#路线图"><img src="https://img.shields.io/badge/status-pre--alpha%2C_M0a_in_progress-7b5fd0" alt="状态：pre-alpha，M0a 进行中"></a>
  <a href="#许可证与致谢"><img src="https://img.shields.io/badge/licence-MIT_OR_Apache--2.0-3f6fd1" alt="许可证：MIT OR Apache-2.0"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/Rust-2024_edition-c2572b?logo=rust&logoColor=white" alt="Rust 2024 版次"></a>
  <a href="PLATFORMS.md"><img src="https://img.shields.io/badge/platforms-Linux%2C_Windows%2C_macOS-2f9a5a" alt="平台：Linux、Windows、macOS"></a>
</p>

<p align="center">
  <a href="#工作原理"><b>工作原理</b></a>&emsp;
  <a href="#路线图"><b>路线图</b></a>&emsp;
  <a href="PLATFORMS.md"><b>平台</b></a>&emsp;
  <a href="plan/README.md"><b>设计文档</b></a>&emsp;
  <a href="CONTRIBUTING.md"><b>参与贡献</b></a>
</p>

> [!IMPORTANT]
> **Syntropy 正在开发中，目前还没有任何可安装的东西**。完整设计在 [`plan/`](plan/README.md) 中，实现工作正在里程碑 M0a 推进。本页描述的一切都是已经设计好、计划要做的内容，而不是今天就能运行的东西。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/memory-pool-dark.svg">
    <img src="docs/media/memory-pool-light.svg" width="860" alt="按同一比例绘制的三台机器。一台有 24 GB VRAM 和 64 GB RAM 的游戏电脑共 88 GB，一台笔记本有 48 GB 统一内存，一台迷你主机有 128 GB。单独哪一台都装不下 142 GB 的模型。三台合计 264 GB，装下模型后还剩 122 GB。">
  </picture>
</p>

一台配有 24 GB 显卡和 64 GB RAM 的游戏电脑、一台有 48 GB 统一内存的笔记本，再加上电视下面一台 128 GB 的迷你主机，单独哪一台都装不下一个 2350 亿参数的混合专家（MoE）模型，这个模型在 4 位量化下有 142 GB。但三台加起来有 264 GB。

Syntropy 把它们当作一台拥有多种内存的计算机。它测量每个 GPU 和 CPU 读取每个存储层级的速度，为整个模型选定一个放置方案，并让计算贴近数据：哪个设备的内存存放某个权重，就由哪个设备用它计算。跨机器时，解码阶段只有激活值在机器之间传输。在一台机器内部，不在 GPU 缓存中的权重由 CPU 在它所在的位置直接计算，而不是每生成一个 token 都经 PCIe 复制一次。

## 功能特性

以下内容都已在设计中写明，但目前都还没有实现。

### 单台机器内部

- **每个存储层级都承担工作**。最热的权重和 MoE 专家放在自适应的 VRAM 缓存中。完整的权重集常驻 RAM，CPU 在那里就地计算缓存未命中的部分。冷专家和大型表在需要时用直接 I/O 从 NVMe 读取。
- **专家在用到之前就已预取**。对下一层专家的预测在路由（router）内核里完成，所以当前层还在计算时，下一层的 NVMe 和 RAM 读取就已经开始。
- **靠实测，不靠查表**。每台机器用 Syntropy 自己的真实内核，探测其 GPU、CPU、硬盘和网络链路的实际性能。放置方案根据这些测量结果构建，而不是根据规格表。

### 跨机器

- **整个模型只有一个放置方案**。一个计划统一决定哪些机器参与、每台机器运行哪段层范围、每个张量组放在哪个层级、由哪个设备计算。它还会记录每个组为什么被放在那里。
- **机器组成一个环**。模型被切成多个流水线阶段（stage），每个阶段把隐藏状态直接交给下一台机器。只有在实测延迟通过门槛的链路上（例如基于 Thunderbolt 5 的 RDMA），才会把单个层拆到多台机器上。
- **局域网内零配置，离开局域网只需一个令牌**。同一网络中的机器会自动发现彼此。你在两边屏幕上确认同一个六位数字码后，新机器即被接纳。离家在外的笔记本用邀请令牌加入，令牌默认只能使用一次。
- **掉了一台机器，答案不会丢**。每个阶段都会保留自己最近发送的内容。当某台运行阶段的机器掉线、而其他机器还有余量时，生成会继续进行，客户端不会收到错误。

### 面向你和你的工具

- **默认精确，除非你另行指定**。会改变输出的捷径（例如跳过未命中缓存的专家）必须主动开启，各有名称，并会列在受其影响的每个响应中。
- **不改变答案的推测解码**。MTP 头或草稿模型先提出若干个 token，模型在一次前向传播中把它们全部校验，接受规则保证输出分布与模型本身一致。
- **你的工具已经在用的 API**。提供 OpenAI 风格和 Anthropic 风格的 API 以及一个原生 API；请求到达哪台机器，就由哪台机器提供服务。
- **全程原生代码**。Rust 编写的系统层，面向 CUDA、HIP、Metal、Vulkan 和 CPU 的原生内核，并以 ggml 作为逐算子的回退方案，扩大硬件覆盖面。

## 初步了解

下面是 [RFC-0051](plan/rfcs/RFC-0051-cli-and-dashboard.md) 中规划的命令行。这些命令目前都还不存在。

```sh
syntropy run                     # 只问一个问题，然后推荐一个装得下的模型
syntropy plan <model> --explain  # 不加载模型，预览放置方案和预测速度
syntropy cluster approve <id>    # 核对六位数字码后，接纳一台局域网内的机器
syntropy invite create           # 为离家在外的笔记本生成一次性令牌
```

按计划，每台机器默认在 `http://127.0.0.1:7433` 提供这些 API，因此现有的 OpenAI 和 Anthropic 客户端及 SDK 无需修改即可使用（[RFC-0050](plan/rfcs/RFC-0050-api-surface.md)）：

```sh
curl http://127.0.0.1:7433/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "stream": true,
       "messages": [{"role": "user", "content": "为什么解码受内存带宽限制？"}]}'
```

## 路线图

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/milestones-dark.svg">
    <img src="docs/media/milestones-light.svg" width="860" alt="从 M0a 到 M4 的七个里程碑。M0a 是单台 Linux 机器上的最小端到端骨架，正在进行中。M1a 是首个预览版，M1b 增加 Windows 和 Apple 芯片，M2 是正式发布，支持局域网内的多台机器，M3 功能完整，M4 是 1.0。">
  </picture>
</p>

设计已经完成：25 份 RFC、98 份 ADR，以及分布在代码六个领域中的 1411 个任务，每个任务都有可测试的验收标准，并注明它实现的 RFC 章节和 ADR。里程碑不设日期。[RFC-0001 §16](plan/rfcs/RFC-0001-system-architecture.md#16-milestones) 定义了每个里程碑的范围和退出标准，[ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md) 规定了每个里程碑发布哪些平台，[路线图](plan/implementation/roadmap.md)按依赖顺序列出了每个里程碑的任务。进度在[项目看板](https://github.com/orgs/syntropyai-hq/projects/1)上跟踪。

<details>
<summary><b>各里程碑交付什么</b></summary>
<br>

| 里程碑 | 交付内容 |
|---|---|
| **M0a** | 单台 Linux 机器上的最小端到端骨架（walking skeleton）：一个稠密模型家族（Llama），以 Q4_K 和 Q8_0 运行在 CPU 和一块 NVIDIA GPU 上，经由真实的运行器（runner）提供 OpenAI 风格的流式聊天；从第一次提交起，一致性测试、精度对齐测试框架以及与模型发布方参考输出的比对就作为合并门禁。只构建和测试，不发布。 |
| **M0b** | 在同一台机器上完成 M0 的其余部分：一个 MoE 模型家族（Qwen-MoE，所有专家都放在 VRAM 中）、经过校准的容差、经过认证的内核选择、首批硬件探测，以及与提交绑定的基准测试框架。不发布。 |
| **M1a** | 在一台配备 NVIDIA GPU 或只有 CPU 的 Linux 机器上实现分层：VRAM、RAM 和 NVMe 三个层级，带预测和预取的专家缓存，单机放置方案，带前缀缓存和会话保存的 KV 分页，以及推测解码。首个公开发布版本，标注为预览版。 |
| **M1b** | 配备 NVIDIA 的 Windows 和 Apple 芯片上的 macOS，以及两者上的纯 CPU 运行。具体范围取决于 M1a 的测量结果。 |
| **M2** | 局域网内的多台机器：发现与配对、加密传输、环形流水线、点对点权重分发和首个仪表盘，另有面向 AMD 和 Intel GPU、基于 ggml 的 Vulkan 预览。这个版本就是正式发布。 |
| **M3** | 功能完整：邀请链接、丢失机器后的恢复、高速链路上的张量并行和专家并行、跨机器的推测解码、原生 HIP 和 Vulkan 内核，以及面向多用户的吞吐模式。 |
| **M4** | 1.0：Tier 2 和 Tier 3 支持级别的平台、带签名的安装包和更新，以及在参考硬件上运行的性能回归门禁。 |

</details>

## 计划支持的平台

每个平台都有一个支持级别（[ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md)），说明它在 1.0 时的支持程度。[PLATFORMS.md](PLATFORMS.md) 有完整的矩阵，包括最低版本、后端，以及每个平台首次发布的里程碑。

<img src="https://img.shields.io/badge/tier_1-supported-2f9a5a" alt="Tier 1：受支持"><br>配备 NVIDIA 的 Linux x86_64、配备 NVIDIA 的 Windows 11 x86_64、Apple 芯片上的 macOS 15+，以及这三者上的纯 CPU 运行。提供预构建的签名版本。精度对齐和性能在真实硬件上测试，任何一项失败都会阻止发布。

<img src="https://img.shields.io/badge/tier_2-known_gaps-3f6fd1" alt="Tier 2：受支持，但有已知缺口"><br>使用 AMD ROCm 的 Linux；AMD、Intel 和 NVIDIA GPU 上的 Vulkan；AMD Strix Halo；GB10 级别的 Linux arm64；Raspberry Pi 5 级别的开发板（CPU）；RDMA 网卡；单机多 GPU。提供预构建版本，强制精度对齐，跟踪性能。已知缺口列在发布说明中。

<img src="https://img.shields.io/badge/tier_3-best_effort-818b98" alt="Tier 3：尽力而为"><br>作为节点的 Android 和 iOS；Intel SYCL；面向 Pascal 和 Volta GPU 的 CUDA 12；WebGPU；Windows 10 22H2；WSL2 下配备 NVIDIA GPU 的 Linux。构建以实验性版本发布。

1.0 不计划支持：Intel 芯片上的 macOS、32 位操作系统、arm64 上的 Windows，以及在浏览器内运行的节点。

## 工作原理

为一个用户生成一个 token，需要把所有被激活的权重和整个 KV 缓存各读一遍。每字节对应的计算量很小，所以解码速度取决于这些字节放在哪里，以及紧挨着它们的处理器读取它们有多快。Syntropy 的设计正是围绕这一点：计算向数据靠拢，数据按测量结果放置。

### 单台机器内部

<p align="center"><img src="docs/media/tiers.svg" width="760" alt="一台机器：VRAM 存放热权重和专家缓存，由 GPU 读取；RAM 存放全部权重，CPU 在那里计算缓存未命中的部分；NVMe 存放冷专家和表，并预取到 RAM 中"></p>

- **VRAM** 存放每个 token 都会用到的部分（注意力、稠密层、共享专家、KV 缓存），以及一个随工作负载自适应的缓存，用来存放最常用的 MoE 专家。
- **RAM** 存放完整的权重集。当某个 token 需要的专家不在 GPU 缓存中时，CPU 就在 RAM 中与 GPU 并行计算它，而不是把它经 PCIe 复制过去。台式机 CPU 读取双通道 DDR5 的速度为 76–122 GB/s，约为 PCIe 4.0 x16 向 GPU 提供带宽的 3–5 倍。
- **NVMe** 存放冷专家和大型查找表，以整个专家为单位按连续区段（extent）存储，并用直接 I/O 读取。下一层预测会在路由器请求之前就发起这些读取。

Syntropy 的层级管理器管理每一个常驻字节，因此留在内存中的内容从不由操作系统的页缓存决定。只有在批量较大时（例如较长的提示词分块），并且只在实测的盈亏平衡点表明复制比就地计算更划算的地方，权重才会以流式方式传到 GPU。

### 跨机器

<p align="center"><img src="docs/media/ring.svg" width="760" alt="三台机器组成一个环，每台运行一段层范围：隐藏状态从一台传给下一台，最后一台采样出 token，把它送回第一台，并流式发送给接收请求的那台机器"></p>

模型按层范围切成多个阶段，每台机器一个阶段，这些阶段组成一个环。第一个阶段对输入做嵌入，每个阶段把自己的隐藏状态直接传给下一个阶段。最后一个阶段采样出 token，沿环送回，并流式发送给接收请求的那台机器。流水线增加的是容量，而不是单个请求流的速度，所以收益来自把数据移到更快的内存上。如果一台慢机器加入环中只会多出一跳，它就会被排除在外。

用张量并行或专家并行把单个层拆到多台机器上，每个 token 的每一层都要付出一次网络往返。Syntropy 只在实测集合通信延迟低到足以让这种做法划算的链路上这样做。实际上这意味着 RDMA 级别的链路。Wi-Fi 永远达不到要求。

集群会选出一台机器担任协调者（coordinator），由它保存记录成员关系和计划的事件日志。它从不接触 token，也不在请求路径上，因此失去它不会让任何请求失败。

> [!TIP]
> 更长、更通俗的解释见 [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)。完整设计在各份 RFC 中：层级与驻留（[RFC-0020](plan/rfcs/RFC-0020-tier-manager-and-residency.md)）、权重 I/O 路径（[RFC-0021](plan/rfcs/RFC-0021-weight-io-path.md)）、专家缓存（[RFC-0023](plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md)）、放置（[RFC-0011](plan/rfcs/RFC-0011-joint-placement-solver.md)）、分布式执行（[RFC-0032](plan/rfcs/RFC-0032-distributed-execution.md)），以及发现与邀请（[RFC-0030](plan/rfcs/RFC-0030-discovery-membership-invites.md)）。

## 默认即精确

在默认设置下，Syntropy 的输出必须在公布的容差范围内，与同一模型在相同量化下的参考前向传播结果一致（[ADR-0019](plan/decisions/ADR-0019-exactness-contract.md)）。

- **参考基准**是 Syntropy 的参考执行器：F32 累加、固定的归约顺序，以及从同一文件精确反量化得到的权重。参考执行器本身又会与模型发布方自己的参考实现比对，因此“精确”既涵盖 Syntropy 的内核，也涵盖 Syntropy 对模型本身的解读。
- **测试方法**是在一个带版本号的语料库上，测量 teacher forcing 下的 top-1 一致率和 KL 散度，语料涵盖聊天、代码、长上下文、工具调用 JSON 和多语言文本。容差按数据类型类别根据测量结果设定，带版本号，并写入每个发布版本。
- **必须通过的**是规划器在精确模式下可能选择的每一条路径：每个内核和回退方案、从单块 GPU 到多机环的每种放置形态、推测解码和普通解码、层级之间的 KV 分页，以及恢复的会话。没有通过的路径就是近似的，不管它原本打算做成什么。
- **捷径必须主动开启并带有名称**，例如 `expert_skip`（不计算未命中的专家）或 `kv_lossy`（KV 以低于模型精度的格式存储）。每个响应都会说明自己的保真度以及启用了哪些捷径，要求 `exact` 的请求永远不会由近似配置来处理。

不承诺不同设备之间输出逐字节相同，因为不同硬件在容差范围内的舍入方式不同。计划提供一个较慢的 `deterministic` 模式，在相同构建和相同设备上得到可重复的输出。容差和精度对齐测试框架的规格见 [RFC-0042](plan/rfcs/RFC-0042-quantization-and-numerical-fidelity.md) 和 [RFC-0060](plan/rfcs/RFC-0060-testing-and-correctness.md)。

## 设计文档

[`plan/README.md`](plan/README.md) 是索引，并给出了建议的阅读顺序。

| 路径 | 内容 |
|---|---|
| [`plan/00-vision-and-scope.md`](plan/00-vision-and-scope.md) | 问题、使用场景、平台矩阵、目标模型、成功标准 SC-1 到 SC-32（全部是目标值），以及“功能完整”的含义 |
| [`plan/decisions/`](plan/decisions/) | ADR。ADR-0000 包含产品准则 D1–D8，ADR-0001 到 ADR-0022 是基础决策，其余按领域编号 |
| [`plan/rfcs/`](plan/rfcs/) | RFC-0001（系统架构），以及 24 份领域 RFC，涵盖放置、层级、分布式执行、计算、产品和质量 |
| [`plan/implementation/`](plan/implementation/README.md) | 六个 epic 文件中的任务、路线图、依赖报告、可追溯性、风险登记册，以及检查它们的脚本 |
| [`plan/open-questions.md`](plan/open-questions.md) | 尚未确定的事项，每一项都注明了由哪个任务或哪项测量来解决 |
| [`plan/glossary.md`](plan/glossary.md) | 每个概念只用一个术语，并列出计划中不使用的同义词 |
| [`plan/research/`](plan/research/) | 依据：关于多设备推理和分层内存推理的笔记、各后端的笔记，以及一个引用了带宽数据来源的硬件成本模型 |

计划中的每个数字都有标注：`[target]`（目标值）、附推导过程的 `[estimate]`（估算值），或引用来源的硬件实测数据。其中没有一个是 Syntropy 的基准测试结果，因为目前还没有可以测的东西。

## 参与贡献

AI 智能体通过执行计划中的任务来构建 Syntropy。一个智能体领取一个任务，在自己的分支上实现它，然后提交一个变更：这个变更要满足该任务的验收标准，并注明它实现的 RFC 章节和 ADR。哪些任务可以开始，由任务图决定。

只有以下各项全部通过，变更才会合并（[ADR-0021](plan/decisions/ADR-0021-development-model-ai-agents-open-source.md)）：

- 构建、lint、格式检查，以及单元测试、属性测试和模拟器测试；
- 精度对齐测试框架和一致性门禁；
- 性能回归门禁（当变更涉及有性能测量的路径时）；
- 依赖与许可证策略、命名检查和任务图检查；
- 一个独立的审查智能体（绝不是编写该变更的那个），它根据任务的验收标准和引用的 RFC 章节检查变更。

精度对齐测试框架和一致性门禁会在 M0 中最先构建，早于它们要检查的内核和模型家族。敏感类别的变更还需要维护者批准：密钥、签名、准入和更新客户端，CI 和发布配置，智能体指令文件，新的模型家族，新的直接依赖，以及任何门禁无法检查或需要豁免门禁的变更。

欢迎外部贡献，适用同样的门禁。[CONTRIBUTING.md](CONTRIBUTING.md) 介绍了 issue、pull request、证据要求和 AI 辅助的工作，[AGENTS.md](AGENTS.md) 是智能体的工作指南。

## 安全

请私下报告漏洞，不要发到公开 issue 中。[SECURITY.md](SECURITY.md) 提供报告途径、响应时间目标，并概述了 Syntropy 将在网络上暴露的内容：局域网发现、配对、邀请链接、它的 API 和它的更新通道。

## 许可证与致谢

Syntropy 采用 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双许可证，由你任选其一。贡献按相同条款接受，并且每次提交都要带有 Developer Certificate of Origin（DCO，开发者原创证书）签署（[ADR-0018](plan/decisions/ADR-0018-licence.md)）。

Syntropy 的内核层以 [ggml](https://github.com/ggml-org/ggml) 为基础。ggml 是采用 MIT 许可证的张量库，提供广泛的后端覆盖，并为每个没有原生内核的算子提供回退实现。每个发布版本都将附带一个自动生成的 `THIRD-PARTY-NOTICES` 文件，列出其中包含的每个组件。Syntropy 不分发任何模型权重，每个模型保留其自身的许可证。
