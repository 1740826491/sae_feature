# 实验交接：HIFER 层级关系、Token-level 前向与昨天的新实验

本文只记录三件事：层级关系重建到底做什么、为什么重新做 token-level 前向，以及昨天讨论后形成的实验主线和当前仓库内容。

## 1. 重新构建的 HIFER 层级关系

### 做什么

目标不是重新训练 Gemma，也不是根据 domain 标签直接制造层级，而是从 SAE feature 的共同激活关系中构建 activation-derived hierarchy：

```text
feature c  ->  feature u
```

其中 `u` 是更常见、更广泛的 feature，`c` 是更少见、更具体的 feature。边表示：当 `c` 激活时，`u` 往往同时激活；但 `u` 激活时不要求 `c` 一定激活。因此方向是从具体/低频 feature 指向一般/高频 feature。

### 怎么做

1. 从 Gemma 2 9B 第 27 层残差流得到 token 表征。
2. 用 Gemma Scope SAE 编码为 sparse feature activation，保留每个 token 的 top-k feature 及激活值。
3. 统计每个 feature 的频率和 feature pair 的共同激活次数。
4. 用非对称条件频率筛选边：要求 `P(u|c)` 足够高、`P(c|u)` 相对较低，同时满足频率/支持度阈值。
5. 对保留边做支持过滤和 transitive reduction，避免把所有传递关系重复画成直接边。
6. 根据保留的有向图计算 depth、节点数量、边数量，并生成 feature label。
7. domain purity 只作为后续 post-hoc 分析：层级构建本身不使用 domain 标签。

### 是否真的重新走前向

分两条数据路径：

- 文档级 HIFER：使用已经保存的文档级 feature 激活统计，重新运行 HIFER 的关系构建和后处理；这一步不是再次运行 Gemma 前向。
- token-level HIFER：重新运行 Gemma 2 9B 第 27 层前向，并保存每个 token 的 SAE top-150 feature 激活，再从这些 token 级记录重建关系。这条路径是真正重新走了模型前向。

文档级结果已经得到多层图：最大深度 6，约 36,766 个节点、约 33,328,250 条边。对应的 depth-domain purity 总体随深度增加：depth 0 为约 0.063，depth 1 为 0.349，depth 2 为 0.553，depth 3 为 0.639，depth 4 为 0.722；depth 5 只有 10 个节点，样本太少。

## 2. 为什么重新做 token-level 前向

此前的文档级统计回答的是“一个 feature 在哪些文档中出现”，但不能严格回答“一个 token 上哪些 feature 同时出现”。这会影响：

- feature pair 的共激活计数；
- 条件概率 `P(u|c)` 与 `P(c|u)`；
- HIFER 的方向性边和递归深度；
- token-level 的 domain、support crossing 和 routing overlap 分析。

因此重新做 token-level 前向的目的，是把关系构建的基本单位从文档改成 token，并保留每个 token 的稀疏 SAE 激活记录。当前输出格式是每个 token 的 `feature_ids` 和 `activation_values`，每个 token 保留 top-150 feature；前向使用 Gemma 2 9B 第 27 层残差流和 Gemma Scope 9B SAE。

这条 token-level 路径不是为了替代文档级结果，而是为了检查：文档级层级是否在更细粒度的 token 共激活上仍然成立，并为后续 E1–E6 分析提供输入。

## 3. 昨天讨论形成的新实验与当前状态

### 讨论得到的关键问题

我和师兄讨论后，问题从“能否画出一棵树”具体化为：

1. 真实 SAE 激活中是否存在稳定的层级/组合结构？
2. 深层 feature 是否比浅层 feature 更 domain-specific？
3. 不同层级的 feature 是否具有不同的 activation-pattern overlap，能否据此做离线分发？
4. 按 activation pattern 分发后，是否能形成比随机分发更有意义的 feature grouping？
5. 这些真实数据统计能否作为后续 synthetic capacity/interference 实验的 overlap 参数？

### 已做的实验

#### A. 文档级 HIFER hierarchy

已完成关系图、depth 统计和 domain purity。结果支持“较深层 feature 平均更 domain-specific”，但不表示所有深层节点都可靠；最深层节点数量很少。

#### B. 文档级 offline routing overlap

按 depth 和专家数 `K∈{2,4,8}`，用 activation pattern 做离线分组。大致结果：

| depth | K=2 | K=4 | K=8 |
|---|---:|---:|---:|
| 0 | 0.006 | 0.465 | 0.688 |
| 1 | 0.664 | 0.706 | 0.661 |
| 2 | 0.493 | 0.586 | 0.563 |
| 3 | 0.556 | 0.466 | 0.382 |
| 4 | 0.279 | 0.279 | 0.230 |

该实验说明不同 depth 的 activation-pattern grouping 强度不同，尤其 depth 1–3 有明显 grouping；但分组负载不均衡，且 depth 4 样本太少，不能单独据此断言“越深越好”。

#### C. overlap-matched synthetic capacity

已经运行初版 synthetic capacity。由于模型容量足够、任务过于容易，各 routing 的 MSE 几乎都接近 0，因此这一版没有形成有效 bottleneck，不能用来证明 routing 的 capacity 优势。后续若继续做，必须降低 expert capacity、提高 feature 数量或提高映射复杂度。

#### D. fixed-assignment capacity

已经补上固定 assignment 的训练脚本并生成独立结果目录，用于补充比较固定分发、随机分发和 assignment corruption。该实验与简单的 overlap-matched ridge 结果不同，重点是观察训练误差、达到目标误差所需步数和 corruption 敏感性。

#### E. token-level 后续分析

token-level 前向已经产生约 1M token 的稀疏激活分片。统计脚本会先从这些分片生成 token-level frequency/coactivation，再运行 HIFER hierarchy。之后自动进入独立的 E1–E4/E6 分析目录，不覆盖文档级结果。

当前自动链路是：

```text
token-level forward
  -> token frequency/coactivation
  -> token-level HIFER hierarchy
  -> E1 train/test split and routing overlap
  -> E2 recursive binary grouping
  -> E3 PCA/dimension scan
  -> E4 context-support crossing
  -> E6 assignment corruption and geometry checks
```

### 文件位置

- 代码仓库：`/Users/suliuguang/Documents/GitHub/sae_feature`
- 主要文档：`docs/`
- HIFER 构建脚本：`scripts/run_hifer_doc_hierarchy.py`
- 文档级 routing：`scripts/run_offline_routing_overlap.py`
- token-level 前向：`scripts/gemma_hierarchy_runner_residpost.py`
- token-level 统计：`scripts/build_token_level_stats.py`
- token-level 自动链路：`scripts/token_pipeline_watch.sh`
- synthetic/capacity 脚本：`scripts/run_overlap_capacity.py`、`scripts/run_fixed_assignment_capacity.py`
- token-level 独立分析：`run_token_properties.py`

阿里云上的运行结果位于：

```text
/mnt/workspace/Hrj/gemma_hierarchy/results/
```

其中：

```text
dclm_1m_9b_hyper_doc_1125/          文档级 HIFER、purity、routing、capacity
dclm_1m_9b_token_activations_1126/  token-level SAE 激活
dclm_1m_9b_token_hyper_stats_1126/  token-level frequency/coactivation
dclm_1m_9b_token_hyper_1126/        token-level HIFER hierarchy
token_properties_1126/               token-level E1–E4/E6 后续分析
```
