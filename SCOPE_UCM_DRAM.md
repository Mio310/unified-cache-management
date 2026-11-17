# SCOPE：UCM DRAM 热度管理算法简版方案

## 一、背景与目标

当前 UCM 采用 **DRAM + SSD 的 Write-through 架构**，SSD 保留 KV Block 的持久化副本，DRAM 作为高速缓存。

我们希望设计一种更合理的 DRAM 缓存管理算法，解决传统 LRU/LFU 无法充分利用 KV Cache 访问特征的问题。

**核心目标：在有限 DRAM 容量下，保留未来复用收益更高的 KV Block，减少 SSD Read 和请求加载延迟。**

参考论文 *The Storage View of KV Cache*，我们认为 KV 具有两种不同的复用规律：

- **Intra-session Reuse：** 同一 Session 内的重复访问，通常体现短期工作集特征。
- **Cross-session Reuse：** 不同 Session 之间的共享访问，可能体现长期共享热点。

因此，算法需要同时考虑 Request Frequency、Session Frequency 和 SSD Load Cost。

---

## 二、SCOPE 简易版算法

SCOPE（Session-aware Cost-based Placement Engine）第一版采用 **Dual Frequency + Cost-aware** 的设计。

### 1. Hotness Score

$$
\boxed{
Score(b)=
\frac{
(\alpha F_{intra}(b)+\beta F_{cross}(b))
\times\Delta C_b
}{
Size_b
}
}
$$

| 参数 | 含义 |
|---|---|
| $F_{intra}$ | Block 在同一 Session 内的重复访问频率 |
| $F_{cross}$ | Block 被不同 Session 访问的频率 |
| $\Delta C_b$ | SSD 读取相比 DRAM 命中节省的成本 |
| $Size_b$ | Block 占用的 DRAM 空间 |
| $\alpha,\beta$ | Intra/Cross 权重，通过实验确定 |

两个频率采用不同的时间衰减系数：

$$
F(t)=F(t_0)e^{-\lambda(t-t_0)}
$$

第一版先验证 Intra/Cross 双频率，Cost-aware 可作为下一步增量实验，避免同时引入过多变量。

### 2. 缓存管理策略

采用简单的 Admission + Eviction 机制：

```text
Block Access
     |
     v
更新 Intra / Cross Frequency
     |
     v
DRAM Hit? ── Yes ──> 返回数据
     |
     No
     |
     v
SSD Load
     |
     v
计算 Candidate Score
     |
     v
与 DRAM Victim Score 比较
     |
     +-- Candidate 更高 --> Admit + Evict Victim
     |
     +-- Candidate 更低 --> 不长期缓存
```

V1 可以引入轻量 Ghost History，记录已淘汰 Block 的近期再次访问，用于识别过早淘汰。

注意：实际释放 DRAM Buffer 时，必须确认 SSD 副本有效且没有未完成的引用或 I/O。

---

## 三、Trace 采集方案

**Trace 的主要目的：通过一次真实运行，生成可以反复用于不同缓存算法实验的数据集。**

因此，需要采集的是与当前 DRAM Hit/Miss 结果解耦的 **Logical Block Access Trace**。

### 1. 最小 Trace 事件

| 事件 | 内容 | 用途 |
|---|---|---|
| `BLOCK_CREATE` | 新 KV Block 产生 | 模拟 Write-through 写入和初始 Admission |
| `BLOCK_PERSIST_READY` | SSD 副本可读取 | 模拟后端可用性 |
| `BLOCK_READ` | Request 需要读取已有 Block | 缓存策略回放的核心输入 |
| `SESSION_COMPLETE` | Session 真正结束 | 生命周期扩展 |

每个 Block 事件尽量包含：

```text
timestamp
event_seq
block_id
request_id
session_id
block_size
block_index
```

额外保存 `cache_namespace`，避免不同模型或 KV 配置的 Block ID 冲突。

### 2. 在哪里采集？

推荐：

**SGLang + UCM External Linker + UCM Cache/Posix**

- SGLang：提供 Request ID、Session ID、Prefix Block 信息。
- UCM Store 入口：记录真实逻辑 READ/CREATE。
- UCM DRAM/SSD：另外采集 Hit/Miss、Read Bytes、Load Latency，用于验证和成本校准。

如果现有 vLLM + UCM 已经稳定运行，也可以由上层业务透传 Session ID，无须为了采集 Trace 强制迁移到 SGLang。

**关键原则：Simulator 不能直接使用 Trace 中的 DRAM Hit/Miss；每个算法都必须根据自己的缓存状态独立计算命中结果。**

---

## 四、实验验证方案

建议按三个阶段完成。

### 阶段一：Trace 采集与正确性验证

使用自定义 Workload Generator 构造四种场景：

1. 同一 Session 高频访问；
2. 不同 Session 共享 Prefix；
3. Session 暂停后重新访问；
4. 一次性长 Prompt 导致缓存污染。

验证 Trace 是否能正确还原 Request、Session、Block 的访问关系，以及 Write-through 写入事件。

**交付物：可按时间顺序回放的 Logical Trace。**

### 阶段二：离线 Cache Simulator

构建一个独立模拟器：

```text
              Logical Trace
                    |
          +---------+---------+
          |         |         |
         LRU       LFU      SCOPE
          |         |         |
          +---------+---------+
                    |
             指标对比
```

对照算法：

| 算法 | 用途 |
|---|---|
| LRU | 基础 Baseline |
| Decayed LFU | 验证访问频率 |
| Window TinyLFU | 强 Baseline，验证抗缓存污染能力 |
| SCOPE-Dual | 验证 Session-aware 双频率 |
| SCOPE-Cost | 验证加入 SSD Cost 的增量收益 |

分别测试 DRAM 容量为全量唯一 KV footprint 的 5%、10%、20%、40%。

核心评价指标：

- DRAM Hit Ratio
- SSD Read Bytes
- Intra/Cross-session Hit Ratio
- Eviction 后的 Reuse
- Estimated Saved I/O Cost
- Metadata Memory Overhead

**交付物：不同算法、不同 DRAM 容量下的性能对比结果。**

### 阶段三：真实 UCM 集成验证

将离线实验表现最好的 SCOPE 版本接入 UCM DRAM 层。

先使用 Shadow Mode 验证决策和开销，再启用真实 Admission/Eviction。

最终比较：

- SSD Read Bytes 是否减少；
- DRAM Hit Ratio 是否提升；
- P95/P99 TTFT 是否改善；
- 算法 CPU 和内存开销是否可接受。

**交付物：UCM 原型实现及端到端性能报告。**

---

## 五、后续执行步骤

1. **确定实验环境与 Trace Schema**  
   确认 SGLang/vLLM 的实际 UCM 接入路径，以及 `request_id → session_id → block_id` 能否关联。  
   **产出：** 统一 Trace 事件定义和埋点位置。

2. **实现 SCOPETraceCollector**  
   先采集逻辑访问，不修改现有缓存策略；补充必要的 DRAM/SSD 物理事件用于校验。  
   **产出：** 真实 KV Block 访问 Trace。

3. **实现 SCOPECacheSimulator**  
   先实现 LRU，验证可重放性；再加入 LFU、TinyLFU 和 SCOPE-Dual。  
   **产出：** 可重复运行的算法评测环境。

4. **验证 Session-aware 是否有增量收益**  
   对比 Request Frequency 和 Dual Frequency，调节时间衰减、权重与 DRAM 容量。  
   **产出：** SCOPE-Dual 的有效性结论。

5. **扩展 Cost-aware 并集成 UCM**  
   加入 SSD Read Cost、Ghost History、Admission/Eviction，随后进行真实压测。  
   **产出：** SCOPE V1 原型及性能数据。

---

## 六、当前最重要的工作

我建议**暂时不要直接开发 SCOPE 的完整 Hotness Controller**。

当前最值得优先投入的是：

**SCOPETraceCollector + SCOPECacheSimulator**

先形成以下实验闭环：

$$
\boxed{
真实访问
\rightarrow Trace
\rightarrow Simulator
\rightarrow 算法对比
\rightarrow 参数优化
}
$$

只要这套基础设施建成，后面调整 Score 公式、增加 Session Completion、Ghost Cache 或 Prefix-aware 策略，都可以快速在同一份数据上验证。

**第一个关键里程碑：使用相同的真实访问 Trace，证明 SCOPE-Dual 是否能够在固定 DRAM 容量下，稳定优于 Decayed LFU 和 Window TinyLFU。**
