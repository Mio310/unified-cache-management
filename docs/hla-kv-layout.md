# HLA 外置缓存方案

Qwen3-Next 从 12 个混合 tensor 收成 4 个：3 个各 12 层 mamba，1 个 12 层 full attention。flash-next 在此基础上再加 compressed、raw_key 和只有 1 层的 PLE，共 7 个 tensor。CUDA 上 Qwen3-Next 按层各拷一页，flash-next 把一层段装进同一个 block。Ascend 上两种模型都按层拷整页，页内 linear 是 conv、ssm 和 padding，full attention 是排满的 K、V。查找先对 full attention 做前缀再核对 linear 与 PLE；dump 写新区间的 full attention，并在 LCM 边界写 state；load 只装命中区间，mamba 写入最后一个非 0 槽。

## Tensor list 怎么变

层的总数仍是 48：12 段 hybrid，每段 3 层 mamba 加 1 层 full attention。变的是这些层怎样放进 KVCacheTensor。

### Qwen3-Next 原先：12 个 tensor

每一段 hybrid 独占一个 tensor。tensor 里面按前向排 3 层 mamba 和 1 层 full attention。12 段就是 12 个 tensor。

```text
T0          T1          T2 … T10       T11
Mamba A0    Mamba A1    3 × Mamba      Mamba A11
Mamba B0    Mamba B1    1 × Full       Mamba B11
Mamba C0    Mamba C1                   Mamba C11
Full 0      Full 1                     Full 11
```

### Qwen3-Next 现在：4 个 tensor

同一种层收进同一个 tensor。3 个 mamba tensor 各放 12 层，full attention 单独一个 tensor 放 12 层。旧 T0 里的三层 mamba 被拆进三个 tensor 的第 0 层，Full 0 成为 full tensor 的第 0 层。

```text
Linear A      Linear B      Linear C      Full attention
Mamba A0      Mamba B0      Mamba C0      Full 0
Mamba A1      Mamba B1      Mamba C1      Full 1
…             …             …             …
Mamba A11     Mamba B11     Mamba C11     Full 11
```

### qwen3.8-flash-next 现在：7 个 tensor

主网络仍是 12 段 hybrid，所以 3 个 linear 和 full attention 仍然各 12 层。full attention 这一组旁边再挂 compressed、raw_key 和 PLE。compressed 与 full attention 各 12 层，物理上落在同一页。raw_key 不进外置存储。PLE 只有 1 层。

```text
Linear A     Linear B     Linear C     Full attention   Compressed   Raw key   PLE
Mamba A0     Mamba B0     Mamba C0     Full 0           Cmp 0        不拷贝    1 层
…            …            …            …                …
Mamba A11    Mamba B11    Mamba C11    Full 11          Cmp 11
```

## 分组

一个 block id 一次只属于一个 group。多个 tensor 可以叠在同一块显存上，拷贝形状可以相同，哈希按 group 分开。

| 角色 | Qwen3-Next | flash-next | 哈希 | 拷贝 |
| --- | --- | --- | --- | --- |
| full attention | 有 | 有 | token 前缀 | 拷 |
| linear | 3 组 | 3 组 | mamba-align state，seed 含 group id | 拷 |
| PLE | 无 | 1 层 | 与 linear 同一套 state hash | 拷 |
| compressed key | 无 | 与 full attention 同页 | 不单独建 key | 随 full attention 一起 |
| raw_key | 无 | 有 | 不哈希，不进 LCM | 不拷 |

`shard_size` 是一个 block hash 下面一份分片的字节数。`shard_index` 是这份分片的编号。整存只有一份，`shard_index = 0`。layerwise 按页拆开，编号用来选第几页。

## CUDA

### Qwen3-Next：LBNHC

tensor 是层在外。`layer_stride = page × num_blocks`。四个 group 叠在同一块显存上，按 `(layer0, size)` 去重。一个逻辑 block 是每一层各取一页，页与页之间隔着 `layer_stride`。

```text
L0 B0 | L0 B1 | L0 … | L1 B0 | L1 B1 | L1 …
```

tensor 里一层占一整段。逻辑 block b 是每一层的第 b 页，不能从 layer 0 连续往下拷。

每一层第 b 页的地址都是 `layer_base + layer_index × layer_stride + block_id × page`。四个 group 叠在同一块显存上，第 i 页同时是三个 linear tensor 和 full attention tensor 的第 i 层。block id 属于哪个 group，这一页里就是那一组的数据。

整存整取把这 12 页收成一个对象。`tensor_size_list = [page] × 12`，`shard_size = block_size = 12 × page`，`shard_index = 0`。一次拷贝带上该 block 每一层的那一页。

layerwise 把这 12 页拆成 12 个 shard，每一行的形状相同：`tensor_size_list = [page]`，`shard_size = page`，`shard_index` 就是页号 0 到 11，`block_size` 仍是 `12 × page`。哈希与整存相同，同一个 block hash 靠 `shard_index` 选页。拷贝地址仍是 `layer_base + i × layer_stride + block_id × page`，长度只有这一页。

第 i 页上是同一段 hybrid 的四层：linear A、B、C 的第 i 层，然后是 full attention 的第 i 层，对应模型层 4i 到 4i+3。四个 tensor 共用这一页，不按 group 再拆行。block id 属于哪个 group，页里就是那一组的数据。dump 和 load 的 block 列表与整存相同：full attention 是命中区间里的那些 block，linear 是 LCM 边界上的那条 state。每个 shard 只搬这些 block 的第 i 页。

这里的逐层是按物理页，不是按四个模型层各拷一份。第 i 页只有 `page` 字节。linear A、B、C 和 full attention 的第 i 层地址相同，是这一页的四套视图，所以一次 load 或 dump 只搬这一页。前向时这四层用各自的 block id，数据落在这一层区域里的不同 block 上，不会把四层的字节拼成 `4 × page`。

调度仍按段推进。第 0 页在 forward 开始前装入，这一段的三层 linear 和 full attention 都读它。vLLM 在 full attention 上等待下一次加载：此时三层 linear 已经写完，保存当前页，并装入下一页，供下一段 linear 使用。最后一页没有下一页可装。

### flash-next：BLNHC

tensor 是 block 在外。一个物理 block 的长度是 `block_stride`，这一段里按 `layer_stride` 排层。linear、full attention、PLE 的 `layer_stride` 都是 `S`，compressed 的是更小的 `s`。大家的 `block_stride` 相同。`block0` 来自 `data_ptr`，已经包含 offset。要读 offset，不能默认 0，也不能再减 `offset % block_stride`。

linear 的一个 block：12 层，每层 `S`。剩余字节是这一页里的 padding，随整页一起拷。

```text
L0 S | L1 S | … | L11 S | pad
```

PLE 的一个 block：只有 1 层有效数据，长度 `S`。后面的 padding 属于这一页。

```text
PLE S | pad
```

三个 linear 各用自己的 `block0`。PLE 也用自己的 `block0`。拷贝都是 `block0 + block_id × block_stride`，长度 `block_stride`。

full attention 与 compressed 共用的一个 block：从两者里更早的 `block0` 起拷一整段 `block_stride`。谁在前由两个 `block0` 的大小决定，差必须小于 `block_stride`。下面是 compressed 在前的一种排法：先 12 层 × `s`，再 12 层 × `S`。

```text
cmp L0 s | … | cmp L11 s | attn L0 S | … | attn L11 S | pad
```

raw_key 在自己的 tensor 上，不哈希、不拷贝、不参与 LCM。

## Ascend

Qwen3-Next 两边的拷贝一样。4 个 tensor 叠在一起，层在外，一个逻辑 block 是每层一页，整存和 layerwise 的 shard 切法也相同。一页里的 conv、ssm、padding 或 7 对 K、V 都跟着这一页整段拷走，不再按组件切开。

### Qwen3-Next

tensor 仍然是层在外。`layer_stride = num_blocks × block_stride`。3 个 linear 和 1 个 full attention 的起点、`size` 相同，是同一块显存上的四套视图。第 i 页同时对应这四个 tensor 的第 i 层。按前向，这一页上是连续的三层 linear 加一层 full attention，例如 0、1、2 与 3。

```text
层 i · B0 | 层 i · B1 | 层 i · … | 层 i+1 · B0
```

tensor 的一层区域里，block 紧挨着排。逻辑 block b 仍是每一层的第 b 页。

一页内部已经按 block 打包，长度就是 `block_stride`。linear 视图和 full attention 视图解释的是同一段字节。

linear 视图的一页：conv `(3, 5120)` 30720 字节，ssm `(24, 128, 128)` 1572864 字节，后面 231424 字节 padding。三者相加等于 1835008。

```text
conv | ssm | pad
```

full attention 视图的一页：一页 K 或 V 是 `(128, 2, 256) × 2` 字节 = 131072。K 紧挨 V。7 对正好铺满 1835008，视图的 `shape[0] = 7 × block 数`。

```text
K | V | K | V | ×7
```

地址是 `block0 + layer_index × layer_stride + block_id × block_stride`。整存的 `tensor_size_list = [block_stride] × 层数`。layerwise 一页一个 shard，`tensor_size_list = [block_stride]`，`shard_index` 是页号，总字节数与整存相同。`block0` 用 `data_ptr`，offset 已经在里面。`data_ptr` 还指着整块显存起点时，`block0 = ptr + offset`。

### flash-next

分组和 CUDA 的 BLNHC 相同：3 个 linear、1 个 PLE、full attention、compressed、raw_key。页的切法与 Ascend 上的 Qwen3-Next 相同，层在外，一页长度 `block_stride`。

linear 一页仍是 conv、ssm、padding。full attention 一页仍是排满的 K、V。compressed 若与 full attention 的 `block0` 相差小于 `block_stride`，就落在同一页里，从更早的 `block0` 拷这一整页，不给 compressed 单独建对象。PLE 用自己的 state hash，从自己的 `block0` 拷；有效数据是它那一层，其余是这一页里的 padding。raw_key 不拷。

```text
compressed | full attn K/V | pad
```

同一页上的两种注意力视图。compressed 相对 full attention 的位置由两个 `block0` 决定，差必须小于一页。

## Lookup

LCM 只算会进出外置存储的 group。primary 是 `self_attn.attn`。

1. 对 full attention 做前缀查找，多组取最短命中，再向下对齐到 LCM。整存至少留 1 个 token 重算。
2. 对每个 linear 和 PLE 做反向查找。state hash 是 `request_hasher(group.seed, UCM_MAMBA_ALIGN_STATE, seq_len, prefix_hash)`。任一 group 缺失，外部命中为 0。

Qwen3-Next 的第二阶段是 3 个 linear。flash-next 再加上 PLE。PLE 在 `state_idx` 处的 block id 为 0 时，这条 state 不写入，整条请求的外部命中变为 0。

## Dump

`token_processed` 从命中长度开始，已命中的前缀不再写。

| 对象 | 内容 | 位置 | 跳过 |
| --- | --- | --- | --- |
| full attention | 区间内每个 block 的前缀 hash | 该 block 的整页 | block id 为 0 |
| linear / PLE | LCM 边界上的一条 state hash | block table 的 state_idx | 该槽为 0，或本段没有落在 LCM 边界 |
| compressed | 不单独写 | 包含在 full attention 那一页里 | flash-next 才有这一段 |
| raw_key | 不写 | 留在 vLLM | 始终 |

Qwen3-Next 在 CUDA 上是每层一页。flash-next 在 CUDA 上是从该 group 的 `block0` 起一整段 `block_stride`；full attention 那一次把 compressed 一起带走。Ascend 上两种模型都是每层一页，页内布局见上面的 tensor 图。layerwise 在这一组 full attention 结束时保存当前页。

## Load

full attention 只装 `[hbm_hit, total_hit)`，重算尾不装。linear 和 PLE 的 hash 取 `total_hit` 这个位置，写入 block table 里从后往前第一个非 0 的槽。dump 用精确下标，load 用这个活槽。

```text
HBM 已有 | 外置 load | 重算尾
```

已命中前缀不再 dump。外置 load 覆盖命中区间。尾部留给本次重算。

Ascend layerwise 用同一个 block hash，`shard_index` 区分页。第 0 页在 forward 前装入，之后每次走到 full attention 装下一页。
