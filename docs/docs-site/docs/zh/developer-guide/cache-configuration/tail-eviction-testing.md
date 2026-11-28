# DRAM 尾部淘汰测试

本页用于在测试环境中验证 Cache Store 的 DRAM 尾部淘汰。功能面向 vLLM 的 Direct 和 Layerwise 路径，包含 MLA；当前不支持 CP、PP、Mamba、Lite 和混合缓存专用 connector。GPU 和 NPU 上的端到端行为需要分别实测，不能仅凭 CPU 单测推断。

## 准备环境和配置

在目标设备的 vLLM 或 vLLM-Ascend 环境中，从包含该功能的同一源码版本**完整重建并安装** UCM：Pipeline 扩展和所有参与的 Store 动态库必须来自同一次构建。建议使用干净的源码检出，按[源码构建说明](../build_from_source.md)设置对应的 `PLATFORM` 后执行 `python -m pip install -v -e . --no-build-isolation`。不要将旧的 Pipeline、Cache 或 Posix 动态库与新版本混用。记录源码提交、设备、驱动、推理框架版本和构建日志。

准备独立、可写的 Posix 缓存目录，并保存以下 YAML：

```yaml
# 根级 Connector 开关，省略时为 false。
enable_dram_tail_eviction: true
use_layerwise: true
enable_metrics: true

ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-tail-test"
      cache_buffer_capacity_gb: 4
```

`cache_buffer_capacity_gb: 4` 只是压测起点，不保证满足所有模型的最小 shard 容量；初始化提示容量不足时按提示增大，并确认宿主机内存及 `/dev/shm` 足够。不要用生产缓存目录。按[快速开始](../../user-guide/quick_start/index.md)通过 `kv_connector_extra_config.UCM_CONFIG_FILE` 指定该文件，并将 vLLM 的 pipeline、prefill context 和 decode context 并行度都设为 1。`enable_dram_tail_eviction` 必须位于 YAML 根级，不放在 `ucm_connector_config` 内。

尾部元数据按共享域定容。若域内共有 `D` 个 DRAM 槽，则 shard 表容量为 `D`，关系节点表容量为 `D + cache_tail_max_batch_blocks × cache_tail_max_inflight_batches`。两项可在 `ucm_connector_config` 配置：

| 配置项 | 含义 | vLLM 默认值 |
| --- | --- | --- |
| `cache_tail_max_batch_blocks` | 单次 Load 或 Dump 的去重关系键上限。配置值需覆盖最长 Load，如果 Dump 会合批，也需覆盖合批后的峰值。 | 8192 和最长序列所需 Store 块数中较大者 |
| `cache_tail_max_inflight_batches` | 域内所有 rank 共用的 Load 和 Dump 在途批次上限，引用释放后归还名额。 | 16 和本地 rank 数两倍中较大者 |

启动日志会打印两项上限和实际分配字节数。元数据需求超过 256 MiB 时初始化失败。批次超过块数上限时返回 `NoSpace`。没有空闲批次名额时，提交会在 `timeout_ms` 期限内等待。

## 最小验证流程

1. **对照组**：用 `enable_dram_tail_eviction: false` 启动服务，记录模型、并行度、DRAM 容量和配置；先发送一批固定的长前缀请求，再重复相同请求。记录输出、日志、命中和内存指标。
2. **实验组**：停止对照组的所有进程，使用独立的空缓存目录和相同的模型、容量、并行度、请求顺序及生成参数，设置开关为 `true` 后重新启动并重放同一批请求。启动或任务报错应作为测试失败记录，不应视为自动回退到旧策略。
3. **命中验证**：确认重复请求确实经过外部 Store 加载，并检查 DRAM 命中指标有变化。如果请求直接命中了设备侧 KV Cache，不能用它证明 DRAM 命中；需要安排其他请求使目标前缀离开设备缓存，再次请求并确认发生外部加载。
4. **容量压力**：使用多个共享长前缀、各自带不同尾部的请求，再持续加入新前缀直到 DRAM 使用量接近容量并出现淘汰字节增量。重复访问早期前缀，记录命中率、淘汰量、错误和延迟，并与对照组比较。固定生成参数，对比缓存复用前后的输出，检查数据正确性。

命中率和淘汰字节只能证明加载或回收活动，不能单独证明具体叶子选择、跨 rank 的完整 block 回收或并发安全。需要验证这些性质时，应结合专门的集成测试或可定位到 block 的调试记录；对不同设备分别执行测试。

## 观察和失败处理

默认配置下可使用项目内置指标集，Prometheus 和 Grafana 的接入见[指标监控](../../user-guide/observability/metrics.md)。可导入 `examples/metrics/grafana_store.json`，重点观察：

| 指标 | 含义 |
| --- | --- |
| `ucm:cache_dram_hit_shards_total` / `ucm:cache_dram_load_shards_total` | 成功加载的 shard 中，来自 DRAM 的比例；分母为零时无命中率。 |
| `ucm:cache_dram_used_bytes` / `ucm:cache_dram_capacity_bytes` | DRAM 缓冲占用和容量。 |
| `ucm:cache_dram_evicted_bytes_total` | 被回收的 ready shard 字节累计量，不能作为完整 block 数量。 |
| `ucm:cache_dram_tail_metadata_used_bytes` / `ucm:cache_dram_tail_metadata_capacity_bytes` | 尾部元数据已用量和分配容量。 |
| `ucm:cache_dram_tail_metadata_gc_total` | 元数据记录 GC 累计量。 |

查看 worker 的 `cache_dram_stats` 日志：每 60 秒输出一次该 worker 的命中和加载 shard 聚合，窗口内没有成功加载时命中率为 `N/A`。`cache_dram_tail_metadata` 日志包含元数据使用量、容量、节点、shard、GC 和容量不足累计数。使用自定义 `metrics_config_path` 时，只有文件中定义的指标才会导出；先确认其中包含所需指标。

设备异步传输的同步失败若导致无法确认 DMA 已结束，worker 会终止以避免继续复用可能仍在使用的内存。发生进程异常退出后，停止并重启**整个共享 Cache 域**的相关进程，确保旧域不再被任何进程使用后再重新创建；不要只重启单个 worker 并继续复用异常域。保留失败前后的日志、配置和指标供分析。
