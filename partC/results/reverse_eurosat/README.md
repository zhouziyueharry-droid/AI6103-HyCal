# EuroSAT 上的 A、B、C 少样本分类对照

本次把 A 的协方差传播和 B 的动态门控迁移到 C 负责的卫星图像数据集 EuroSAT，再与 C 已记录的 75% 协方差共享结果对照。5-shot 时 A 比 C 高约 0.15 个百分点，10-shot 时 C 比 A 高约 0.22 个百分点。三个种子的均值支持“两者接近”；这两档的 B 准确率都低于 A、C。

| 每类标注图片 | A · 协方差传播 | B · 动态门控 | C · 75% 协方差共享 |
|---|---:|---:|---:|
| 1 | 未评估 | 52.78% | 52.78%（余弦回退） |
| 5 | **79.68%** | 73.87% | 79.53% |
| 10 | 86.18% | 82.62% | **86.40%** |

## 实验过程

1. **固定考题。** EuroSAT RGB 有 10 个地表类别。采用 CoOp/Zhou 划分，训练、验证、测试分别有 13,500、5,400、8,100 张图。每个 shot 数都按 C 已保存的种子 0、1、42 读取训练样本，逐一核对文件名、标签和每类张数。A、B 使用相同的测试图片顺序。[九份抽样清单](../../indices/)
2. **提取特征。** 本机 RTX 5060 Ti 使用冻结的 OpenAI CLIP ViT-B/16、FP32 和确定性的评估变换。每个入口各自编码全部测试图和各次抽样涉及的训练图，共 8,554 张。两者使用同一份编码函数、权重和类别文本，提示词格式为 `a photo of a {类别名} satellite location.`。
3. **用各自固定的参数分类。** A 沿用 Flowers102 选定的投影模式，把共享的图像特征协方差映射到每个候选类的图文表示空间，再与本类协方差各取 50%；岭正则为 0.0001，距离分数权重为 1。其 1-shot 公式无法估计类内变化。B 直接调用 B 同学提交的 `Low-Margin Strong` 重建函数。它根据每张图余弦分数前两名的差距调整统计距离权重。A、B 均沿用已确定的参数在 EuroSAT 测试集上评分。[A 的冻结配置](../../../results/part_a_round2/propagation/selection.json) · [B 的实现和参数](member_b_reconstructed/RESULTS.md)
4. **计分。** 对每个种子的 8,100 张测试图计算 Top-1 准确率，再对三个种子取算术平均。下表的提升先按同一种子的未舍入结果相减，再取平均。A、B 的准确率可从逐图预测复算。C 的 5/10-shot 引自组员此前同划分、同抽样的固定参数消融表；1-shot 引自其余弦回退记录。[C 的固定参数表（D 减 B）](../param_search/ablation_shrink.md) · [C 的单域记录](../single_domain/eurosat/summary.md)

## 各自的改动贡献了多少

| 对照 → 改进 | 5-shot 准确率 | 配对提升 | 10-shot 准确率 | 配对提升 |
|---|---:|---:|---:|---:|
| A：同配置、本类协方差 → 加入传播后的共享协方差 | 76.67% → 79.68% | +3.01 pp | 83.47% → 86.18% | +2.71 pp |
| B：B 包的 Original HyCal → 最终动态门控 | 71.37% → 73.87% | +2.50 pp | 79.81% → 82.62% | +2.80 pp |
| B：统一增强权重 ×1.5 → 按图片动态调权 | 73.60% → 73.87% | +0.27 pp | 81.93% → 82.62% | +0.69 pp |
| C：相同打分参数，共享比例 0% → 75% | 76.60% → 79.53% | +2.93 pp | 83.49% → 86.40% | +2.92 pp |

pp 表示百分点。第一张表比较分类准确率；第二张表衡量各方法相对指定对照的变化。B 的 Original 在 5-shot 只使用余弦分数，最终方法还降低了启用统计距离的门槛。要衡量按图片调权本身的贡献，应看统一增强那一行。

## 如何复现

从仓库根目录运行。使用 Python 3.11+、支持 CUDA 的 PyTorch 和匹配版本的 torchvision，并安装 NumPy、Pillow、ftfy、regex、tqdm。将以下三个文件以表中名称放入 `artifacts/paired_inputs/`，运行入口会核对哈希。

| 文件 | 来源 | 预期校验值 |
|---|---|---|
| `EuroSAT_RGB.zip` | [EuroSAT 作者的 RGB 压缩包](https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1) | MD5 `f46e308c4d50d4bf32fedad2d3d62f3b` |
| `split_zhou_EuroSAT.json` | [此次使用的 CoOp 划分副本](https://raw.githubusercontent.com/linzhiqiu/cross_modal_adaptation/main/splits/split_zhou_EuroSAT.json) | SHA-256 `541503a77d0b26b735d85ec7a3e672804fe0aa466c2e5f07da772cc6c0af8ad2` |
| `ViT-B-16.pt` | [OpenAI CLIP 权重](https://openaipublic.blob.core.windows.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |

```sh
python partC/run_reverse_eurosat.py --include-gate
python partC/run_member_b_eurosat.py
```

第一条命令复现 A 及早期门控对照，写入本目录的[逐种子成绩](runs.csv)和[逐图预测](predictions.csv)。其中 `A_local` 和 `A_propagation` 对应 A 的两列配置；`B_gate_reimplementation` 是 C 按 B 的思路写过的早期版本，单独保留以便追溯。

第二条命令复现主表中的 B，写入子目录的[逐种子成绩](member_b_reconstructed/runs.csv)和[逐图预测](member_b_reconstructed/predictions.csv)。B 的评分函数来自成员按原医学实验记录重建的交付包。这里报告的是这份代码在 EuroSAT 上的实测结果；原 OrganAMNIST 服务器脚本和逐图预测尚待核对。
