# Gemma 2 9B SAE feature hierarchy：实验交接说明

这份文档说明本仓库中保存的代码和结果，以及如何复现实验。

## 实验目的

使用 DCLM 文本让 Gemma 2 9B 做前向，在与 SAE 训练一致的表征位置取得 SAE feature activation，然后：

1. 用 CAR 的方向性条件概率统计 constituent → composite 的两层 feature hierarchy；
2. 保存每个 document 的稀疏 feature ID 集合；
3. 对高层 feature 的 document 激活模式做分类：在相似 document 上激活的 feature 归为同一类。

这里的分类不是按激活频率等量切分，也不是按 feature ID 随机切分。它使用 feature 在真实 document 上的激活向量相似度。

## 模型、SAE 与表征

- Base model：`google/gemma-2-9b`
- SAE：Gemma Scope 9B residual SAE，width 131K，`average_l0_56`
- Representation：`resid_post`
- Hook：layer 28 输入，即 layer 27 输出后的 residual representation
- SAE feature activation：Top-K=150
- 数据：DCLM，快速验证版本约 1.01M tokens，4,248 documents

SAE 权重不放进 GitHub；服务器上的 SAE 路径为：

```text
/mnt/workspace/Hrj/gemma_hierarchy/models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz
```

## 仓库文件结构

```text
scripts/
  car_smoke_exact.py
  gemma_hierarchy_runner_layer28_input_cardoc.py
  gemma_hierarchy_runner_manual_residpost.py
  cluster_activation_patterns.py
  cluster_relation_nested.py
  build_joint_relation_classes.py

results/gemma2-9b-layer27-1m-residpost-top150/
  car_exact_edges.csv
  car_exact_labels.csv
  car_exact_complete.json
  strict_feature_classes/
  activation_pattern_clusters/
  relation_nested_classes/
```

`doc_features_rank*.npy` 是原始 document-feature 稀疏激活文件，体积较大，保留在阿里云共享目录中，不提交到 GitHub。它们是后处理和聚类的输入。

## 前向与稀疏激活

前向脚本读取 DCLM 数据，加载 Gemma 2 9B 和对应 SAE，在 layer 28 输入处取得 `resid_post`。对每个 document，只保存 Top-150 feature ID：

```text
doc_features_rank0.npy
...
doc_features_rank7.npy
```

每个 `.npy` 文件是 object array；每个元素对应一个 document 的稀疏 feature ID 数组。

## CAR hierarchy

对 feature `f`、`g` 统计：

```text
P(g | f) = count(f and g) / count(f)
P(f | g) = count(f and g) / count(g)
```

在方向性 containment、反向 containment、频率比和 joint containment 条件下，生成：

```text
parent constituent feature -> child composite feature
```

本轮结果：

- active features：67,839
- accepted composites：14,208
- directed edges：110,524
- maximum depth：1，即 depth 0 constituent + depth 1 composite 两层

仓库中的结果：

```text
results/gemma2-9b-layer27-1m-residpost-top150/car_exact_edges.csv
results/gemma2-9b-layer27-1m-residpost-top150/car_exact_labels.csv
```

## 严格组成类

如果两个高层 feature 的底层 constituent 集合完全相同，就归入同一个严格组成类。例如：

```text
E <- {A, B, C}
F <- {A, B, C}
G <- {A, B, C}
```

则 E、F、G 共享同一个组成类。第一轮共有 930 种 constituent signature；其中最大的重复 signature 包含 13,277 个高层 feature 和同一组 8 个底层 feature。

对应文件：

```text
results/gemma2-9b-layer27-1m-residpost-top150/strict_feature_classes/strict_class_members.csv
results/gemma2-9b-layer27-1m-residpost-top150/strict_feature_classes/strict_class_summary.csv
```

## document 激活模式分类

为了在同一结构类内部得到更细的类别，使用高层 feature × document 的二值激活矩阵：

```text
feature_i -> [document_1 是否激活, ..., document_4248 是否激活]
```

处理步骤：

1. 从 `doc_features_rank*.npy` 构造 feature-document incidence matrix；
2. 使用 TF-IDF 降低普遍 document 的影响；
3. 使用 cosine similarity 表示激活模式相似度；
4. 用 `MiniBatchKMeans` 测试 K=4、8、12、16；
5. 使用 cosine silhouette 选择候选中最好的 K。

本轮测试结果：

| K | Cosine silhouette | 类大小 |
|---:|---:|---|
| 4 | 0.0075 | 3764, 3605, 3442, 2466 |
| 8 | 0.0057 | 1196, 2574, 1714, 2050, 985, 1370, 817, 2571 |
| 12 | 0.0048 | 750, 1733, 1744, 448, 1745, 1607, 286, 612, 703, 809, 1067, 1773 |
| 16 | 0.0026 | 更细但分离度更弱 |

因此当前保存 K=4 的分类标签。silhouette 较低，说明四类是可用于第一轮数据生成的 activation-pattern labels，但不是强语义边界。

结果文件：

```text
results/gemma2-9b-layer27-1m-residpost-top150/activation_pattern_clusters/cluster_labels.csv
results/gemma2-9b-layer27-1m-residpost-top150/activation_pattern_clusters/cluster_summary.csv
```

其中 `cluster_labels.csv` 的字段为：

```text
feature,cluster,activation_docs
```

## 底层到顶层的关系感知分类

最终分类保留 CAR 的真实底层到顶层映射，而不是只对高层 feature 独立聚类：

1. 对 CAR 中实际出现的底层 parent/constituent features 按 document 激活模式分成 3 个底层大类；
2. 每个高层 child/composite 读取它实际连接的底层 parent 类别，得到 `parent_cluster_signature`；
3. 只在对应的底层大类内部，把高层 feature 按 document 激活模式再分成两个子类。

逻辑形式是：

```text
bottom_0 -> bottom_0_top_0, bottom_0_top_1
bottom_1 -> bottom_1_top_0, bottom_1_top_1
bottom_2 -> bottom_2_top_0, bottom_2_top_1
```

实际数据中如果某个底层大类没有足够的 CAR child，不会人为伪造空的顶层类别。

## 复现后处理

在包含 `doc_features_rank*.npy` 的目录中运行 CAR：

```bash
python scripts/car_smoke_exact.py \
  --out /path/to/output \
  --minimum-count 50 \
  --containment 0.85 \
  --reverse 0.85 \
  --frequency-ratio 1.10 \
  --joint-containment 0.75 \
  --minimum-constituents 2 \
  --maximum-constituents 8 \
  --max-base-candidates 4096
```

然后按严格组成类生成 `strict_feature_classes`，再用 document 激活模式聚类：

```bash
python scripts/cluster_activation_patterns.py \
  --input /path/to/output \
  --features /path/to/output/strict_feature_classes/strict_class_summary.csv \
  --output /path/to/output/activation_pattern_clusters \
  --k 4 8 12 16
```

生成关系感知的底层—顶层标签：

```bash
python scripts/cluster_relation_nested.py \
  --input /path/to/output \
  --edges /path/to/output/car_exact_edges.csv \
  --output /path/to/output/relation_nested_classes \
  --bottom-k 3 \
  --top-sub-k 2

python scripts/build_joint_relation_classes.py \
  --edges /path/to/output/car_exact_edges.csv \
  --top-labels /path/to/output/relation_nested_classes/top_cluster_labels.csv \
  --output /path/to/output/relation_nested_classes
```

## 结果如何使用

- 如果需要 CAR 的层级关系，读取 `car_exact_edges.csv`；
- 如果需要严格组成关系，读取 `strict_class_members.csv`；
- 如果需要给高层 feature 打 activation-pattern 类别标签，读取 `activation_pattern_clusters/cluster_labels.csv`；
- 后续生成数据时，优先使用 `cluster` 作为 feature class label，同时保留原始 `feature` ID 和 CAR 的 constituent/composite 标签。

## 当前结论

本轮已经完成正确 SAE 表征的前向、稀疏激活保存、CAR 两层 hierarchy 统计，以及基于 document 激活相似度的 feature 分类。模型权重、SAE 权重和原始大规模 activation 文件不提交 GitHub；GitHub 仓库保存可复现代码、压缩后的结构结果、分类标签和本说明。

## 底层—顶层联合关系类

为了直接挑选一个完整的 feature group，最终还生成联合类。一个联合类同时包含：

```text
底层 constituent features + 对应的顶层 composite features
```

例如：

```text
joint_0000 = {A, B, C} + {E, H}
joint_0001 = {A, B, C} + {F}
```

联合类按两部分确定：

1. 顶层 feature 的真实 CAR parent signature；
2. 顶层 feature 所属的嵌套 activation-pattern label。

因此不会把没有直接 CAR 关系的底层和顶层 feature 强行放在一起。使用时直接读取：

```text
results/gemma2-9b-layer27-1m-residpost-top150/relation_nested_classes/joint_relation_members.csv
results/gemma2-9b-layer27-1m-residpost-top150/relation_nested_classes/joint_relation_summary.csv
```

`joint_relation_members.csv` 中：

- `role=bottom_constituent`：联合类中的底层 feature；
- `role=top_composite`：联合类中的顶层 feature；
- `nested_top_cluster`：顶层子类标签。

后续生成数据时，可以按 `class_id` 选择一个完整的底层—顶层 feature group。

