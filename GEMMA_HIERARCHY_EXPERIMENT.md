# Gemma 2 9B SAE Feature Hierarchy and Activation-Pattern Classes

## 1. 实验目标

本实验的目标不是训练新的 SAE，也不是解释每个 feature 的语义，而是：

1. 让 DCLM 文本数据通过 Gemma 2 9B 的前向计算；
2. 在 SAE 对应的正确表征位置取得 SAE feature activation；
3. 用旧 CAR 逻辑统计 feature 之间的方向性组成关系；
4. 得到两层的 feature hierarchy：底层 constituent feature 和高层 composite feature；
5. 在高层 feature 内部，进一步按照 document 激活模式相似度进行分类。

最终分类的基本原则是：如果两个 feature 在相似的 document 上被激活，它们就更可能属于同一类。这里使用的是 activation-pattern clustering，而不是按激活频率人为分段。

## 2. 模型与 SAE

- Base model: `google/gemma-2-9b`
- SAE: Gemma Scope 9B residual SAE
- SAE path:

```text
/mnt/workspace/Hrj/gemma_hierarchy/models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz
```

- Representation: `resid_post`
- Hook location: layer 28 的输入位置，也就是 layer 27 输出后的 residual representation
- SAE width: 131,072 features
- Forward precision: BF16-compatible model execution
- Feature activation selection: Top-K=150

这里必须使用与 SAE 训练对应的 `resid_post` 表征。旧实验使用的 `post_attention_layernorm` 是另一种 representation，不能与当前 Gemma Scope `pt-res` SAE 混用。

## 3. 数据与前向阶段

本轮使用 DCLM 数据，先处理约 1M tokens 作为快速验证版本。

前向阶段保存的不是连续 activation tensor，而是每个 document 的稀疏 feature ID 集合：

```text
doc_features_rank*.npy
```

每个 document 对应一组被 Top-K 选中的 SAE feature ID。后续 CAR 统计只读取这些稀疏 document-feature 记录，不需要重新加载模型或重新跑前向。

本轮数据规模：

- documents: 4,248
- total tokens: 约 1.01M
- active high-level features: 13,277（最大关系类中的 composite features）

## 4. CAR 方向性关系

对 feature `f` 和候选 feature `g`，使用 document 共现统计：

\[
P(g\mid f)=\frac{N(f,g)}{N(f)}
\]

以及反向条件概率：

\[
P(f\mid g)=\frac{N(f,g)}{N(g)}
\]

当一个 feature 稳定地包含在另一个 feature 的出现集合中，同时反向包含关系较弱时，形成方向性组成关系。

当前输出边的语义为：

```text
parent constituent feature  ->  child composite feature
```

第一轮 CAR 结果：

- active features: 67,839
- accepted composites: 14,208
- directed edges: 110,524
- maximum depth: 1

`max_depth=1` 表示当前是两层结构，而不是只有一层：

```text
depth 0: constituent / carbon feature
depth 1: composite / lambda feature
```

CAR 原始文件：

```text
/mnt/workspace/Hrj/gemma_hierarchy/results/dclm_1m_9b_residpost_top150_car_A/car_exact_edges.csv
/mnt/workspace/Hrj/gemma_hierarchy/results/dclm_1m_9b_residpost_top150_car_A/car_exact_labels.csv
```

## 5. 严格组成类

先按“高层 feature 的底层 constituent 集合完全相同”定义严格类。例如：

```text
E <- {A, B, C}
F <- {A, B, C}
G <- {A, B, C}
```

则 `{A, B, C, E, F, G}` 属于同一个严格组成类。

严格统计结果：

- 14,208 个高层 feature
- 930 种不同的 constituent signature
- 3 个 signature 被至少两个高层 feature 重复使用
- 最大重复类包含 13,277 个高层 feature
- 该最大类共享同一组 8 个底层 feature

严格类输出：

```text
/mnt/workspace/Hrj/gemma_hierarchy/results/dclm_1m_9b_residpost_top150_car_A/strict_feature_classes/strict_class_members.csv
/mnt/workspace/Hrj/gemma_hierarchy/results/dclm_1m_9b_residpost_top150_car_A/strict_feature_classes/strict_class_summary.csv
```

严格组成类非常保守，因此大量高层 feature 的组成 signature 只出现一次。它适合确认结构关系，但不适合产生很多可用类别。

## 6. 最终采用的细分类：document 激活模式聚类

为了在最大关系类内部得到更细的 feature 类别，不再按激活频率排序切分，也不再按某个单独底层 feature 进行硬分组。

对每个高层 feature 构造一个 document activation vector：

```text
feature_i -> [是否在 document_1 激活, ..., 是否在 document_4248 激活]
```

然后：

1. 用 TF-IDF 对 document 维度进行加权，降低所有 feature 都容易激活的 document 的影响；
2. 使用 cosine similarity 衡量两个 feature 的激活模式相似度；
3. 使用 MiniBatch-KMeans 进行候选分类；
4. 比较多个 K 值的 cosine silhouette score；
5. 选择候选 K 中分离度最高的结果。

测试结果：

| K | Cosine silhouette | 类大小 |
|---:|---:|---|
| 4 | 0.0075 | 3764, 3605, 3442, 2466 |
| 8 | 0.0057 | 1196, 2574, 1714, 2050, 985, 1370, 817, 2571 |
| 12 | 0.0048 | 750, 1733, 1744, 448, 1745, 1607, 286, 612, 703, 809, 1067, 1773 |
| 16 | 0.0026 | 多个更小类别 |

在已测试的候选 K 中，K=4 的分离度最高，因此当前采用 4 类。

需要谨慎解释：silhouette 只有 0.0075，说明这些 activation-pattern 类之间的自然分界较弱。它们可以作为第一轮 feature label 使用，但不能解释成四个非常清晰、互不重叠的语义类别。

聚类输出：

```text
/mnt/workspace/Hrj/gemma_hierarchy/results/dclm_1m_9b_residpost_top150_car_A/activation_pattern_clusters/cluster_labels.csv
/mnt/workspace/Hrj/gemma_hierarchy/results/dclm_1m_9b_residpost_top150_car_A/activation_pattern_clusters/cluster_summary.csv
```

`cluster_labels.csv` 中每一行对应一个高层 feature，包含：

- `feature`: SAE feature ID
- `cluster`: activation-pattern cluster ID
- `activation_docs`: 该 feature 被激活的 document 数量

## 7. 当前结果的正确解释

当前实验提供两种互补的标签：

### 7.1 结构标签

来自 CAR：

```text
constituent -> composite
```

它回答“哪些 feature 组成了哪些高层 feature”。

### 7.2 激活模式类别标签

来自 document activation-pattern clustering：

```text
feature_i 和 feature_j 是否在相似 document 上激活
```

它回答“哪些高层 feature 在数据分布上表现得更相似”。

后续生成数据时，可以使用第二种 cluster label；如果需要保留层级信息，则同时保存 CAR 的 constituent/composite 关系。

## 8. 不采用的方法

以下方法不作为当前最终分类：

1. 按 feature activation frequency 直接等量三分；这只能保证类别大小接近，不能说明 feature 之间真的相似。
2. 按最大底层 feature 进行分类；本轮测试中会把全部 13,277 个高层 feature 分到同一个主导底层类别，区分度不足。
3. 直接使用无向连通分量；它会通过关系链把大量 feature 合并成一个超大组件。
4. 随机删减最大类中的 feature；这会改变样本数量，但不会产生新的结构信息。

## 9. 结论

本轮已经完成：

1. Gemma 2 9B 的正确 SAE 表征前向；
2. 稀疏 document-feature activation 保存；
3. CAR 两层方向性 feature hierarchy；
4. 严格 constituent signature 统计；
5. 基于 document activation-pattern similarity 的细粒度 feature 分类。

当前最适合作为后续数据生成输入的是：

```text
activation_pattern_clusters/cluster_labels.csv
```

它使用的是 feature 在真实 DCLM documents 上的激活相似性，而不是人为按照频率切分。
