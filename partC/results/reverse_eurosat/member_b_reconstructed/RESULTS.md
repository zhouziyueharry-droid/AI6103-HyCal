# B 的 Low-Margin Strong 在 EuroSAT 上的结果

在 C 的 EuroSAT 测试集上，B 提供的最终门控函数在 5-shot 和 10-shot 下均超过其原始 HyCal 对照。下表为三个种子的平均 Top-1 准确率，差值用未舍入结果计算。

| 每类样本数 | B 自带 baseline | 统一加权 ×1.5 | B 动态门控 | 相对 baseline（百分点） | 相对统一加权（百分点） | A 的协方差传播 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 52.78% | 52.78% | 52.78% | 0.00 | 0.00 | 未评估 |
| 5 | 71.37% | 73.60% | 73.87% | +2.50 | +0.27 | 79.68% |
| 10 | 79.81% | 81.93% | 82.62% | +2.80 | +0.69 | 86.18% |

A 和 B 共用 EuroSAT 的 8,100 张测试图、CLIP ViT-B/16、相同的类别提示词及种子 0、1、42 的固定支持集。已核对两份逐图记录的图片 ID 和标签，并复算准确率。A 列取自[此前逐种子结果](../runs.csv)。两种方法各自使用冻结的算法参数。

B 的提交包标注为依据 OrganAMNIST 实验记录重建。此处直接调用包内的 [method_scores](../../../member_b_reconstructed/query_gated_hycal.py)，源码 SHA-256 为 `b3ee390c64af29a22a354cf00ca10bda767885eec71864adf16981377f7c1f88`。本次用本地 RTX 5060 Ti 计算 EuroSAT 特征和 HyCal 分数，沿用 B 的 sum fusion、类与背景正则化系数 0.001、样本门槛 10 和 sigmoid 尺度 5。Relaxed 分支从每类 3 张样本开始引入统计分数。统一加权和动态门控均使用同一 Relaxed 分支，比较这两列可衡量逐图调权带来的额外收益。训练和参数选择均未使用 EuroSAT 测试图。

从仓库根目录运行 `python partC/run_member_b_eurosat.py`。[逐种子结果](runs.csv) · [逐图预测](predictions.csv)
