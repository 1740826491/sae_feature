# 假设三：强 common 与其他 leaf 对目标 leaf 的训练干扰

## 研究问题

固定一个 target leaf，分别加入更强、更常见的 common feature 和其他 leaf feature，判断长尾 target leaf 的学习损失来自哪一种干扰，以及两种干扰是否存在额外交互。

四个条件严格复用 Square Output 的 teacher、输入维度、学生结构、优化器和随机种子，只改变训练样本中出现的 component：

| 条件 | 样本中包含的 component | 目的 |
|---|---|---|
| `T` | target leaf | 无额外干扰的基线 |
| `T+C` | target leaf + common/strong component | 测量强 common 的干扰 |
| `T+O` | target leaf + other leaves | 测量其他 leaf 的干扰 |
| `T+C+O` | target leaf + common + other leaves | 测量联合干扰 |

这里的 `target` 是 `T20_0`，`common` 是 `B2_0`，`other leaves` 是 `T20_1、T21_0、T21_1` 的 feature 池。聚类定义没有重新计算，沿用 Square Output 原来的 decoder-similarity 聚类标签。

## 数据和 teacher

每个 component 从对应 feature 池随机抽取一个 feature，使用它的 3584 维 SAE decoder 方向，并乘以独立的 `Uniform[0.5, 1.5]` 强度。输入为所有 component 的和：

\[
x = u_T + u_C + u_O,
\]

缺失的 component 取零。teacher 对每个 component 使用 Square Output 中同一套映射：

\[
y_g = T_g(u_g),\qquad y=\sum_g y_g.
\]

因此四组之间没有改变 feature 的语义、幅值分布或 teacher 映射，只改变同时出现的干扰来源。

## 学生模型和指标

学生是单个共享的低秩线性映射：

\[
3584 \rightarrow 300 \rightarrow 3584,
\]

使用 Square Output 原来的 AdamW、学习率 `0.001`、`weight_decay=0`、batch size `384`、6000 steps 和 seeds `11, 22, 33`。

每个训练完成后，在同一套 target-only 测试分布上测量：

\[
L_T = \frac{\sum_n\|\hat y(u_T)-T_T(u_T)\|^2}
                 {\sum_n\|T_T(u_T)\|^2}.
\]

用三个 seed 的均值计算：

\[
\Delta_C = L_{T+C}-L_T,
\]

\[
\Delta_O = L_{T+O}-L_T,
\]

\[
I_{C,O}=L_{T+C+O}-L_{T+C}-L_{T+O}+L_T.
\]

其中 `I_{C,O}` 是交互项：正值表示联合干扰超过两个单独干扰的加和，负值表示没有额外正协同干扰。

## 正式结果

| 训练条件 | target-only NMSE |
|---|---:|
| `T` | 0.09334 |
| `T+C` | 0.15442 |
| `T+O` | 0.13495 |
| `T+C+O` | 0.17324 |

由此得到：

```text
ΔC       = +0.06108
ΔO       = +0.04161
I(C,O)   = -0.02279
```

## 当前结论

在这套 Square Output、共享 rank-300 学生中：

1. common/strong feature 对 target leaf 的干扰大于其他 leaf；
2. common 的额外 target loss 约为 `0.0611`，其他 leaf 约为 `0.0416`；
3. 两者同时出现时 target loss 为 `0.1732`；
4. 交互项为负，因此目前没有证据表明 common 与 other leaves 会产生额外的正协同干扰。

这回答的是共享参数空间中的干扰问题，不等同于 routed MoE 的路由收益结论。

## 复现

代码：[`run_interference_square.py`](../../scripts/run_interference_square.py)

```bash
python scripts/run_interference_square.py \
  --out docs/square_output/interference_formal_1125
```

正式实验的 12 个 `result.json` 位于 [`interference_formal_1125/`](./interference_formal_1125/)。每个结果目录还包含对应的 `curve.jsonl` 和服务器上的 checkpoint；checkpoint 不提交到 GitHub。

本实验仍需要外部 SAE 权重和聚类 CSV。它们不是本仓库重新训练出来的 ablation 聚类，而是 Square Output 原实验使用的 Gemma 2 9B 第 27 层 SAE decoder 及其 `B2/T20/T21 labels_k2.csv`。
