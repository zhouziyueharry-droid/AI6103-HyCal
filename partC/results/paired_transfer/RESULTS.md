# C 的协方差共享方法在 Flowers102 和 OrganAMNIST 上的配对实验

每类用 5 或 10 张标注图片时，两组数据的 Top-1 准确率都提升，且每个随机种子都观察到了提升。最大提升出现在 OrganAMNIST 的 10-shot 条件下，平均增加 6.02 个百分点。

| 数据集 | 每类样本数 | Baseline | 75% 共享 | 平均变化（百分点） | 变化的样本标准差（百分点） | 提升的种子数 |
|---|---:|---:|---:|---:|---:|---:|
| Flowers102 | 1 | 70.35% | 70.35% | 0.00 | 0.00 | 0/3 |
| Flowers102 | 5 | 93.26% | 95.41% | +2.15 | 0.31 | 3/3 |
| Flowers102 | 10 | 95.02% | 97.17% | +2.15 | 0.36 | 3/3 |
| OrganAMNIST | 1 | 45.47% | 45.47% | 0.00 | 0.00 | 0/3 |
| OrganAMNIST | 5 | 59.37% | 64.30% | +4.92 | 0.59 | 3/3 |
| OrganAMNIST | 10 | 65.61% | 71.63% | +6.02 | 1.18 | 3/3 |

## 实验条件

- Baseline 为本次消融的对照组。采用 C 的 HyCal 实现，每类独立估计协方差（α = 0）。改进版将每类的协方差与同次抽样中所有类别合并估计的类内协方差按 25% 和 75% 混合（α = 0.75）。两组的分类阈值、融合和正则化参数相同。
- 冻结 OpenAI CLIP ViT-B/16，在本机 RTX 5060 Ti 上用 FP32 对图像和文本编码一次。两组共用特征、标注样本、文本提示和完整测试集。分类评分沿用 C 的 CPU 实现。固定 sum fusion、weighted scoring、n_min_p = 2、scale = 1、λ_reg = λ_bg = 0.0001。
- Flowers102 有 102 类，沿用 A 的训练划分、样本索引和 2,463 张测试图。OrganAMNIST 有 11 类，采自官方 224×224 版训练集，在官方 17,778 张测试图上评估；灰度图复制为 RGB。花卉提示词为 `a photo of a {classname}.`，器官提示词为 `a photo of a CT scan of {name}.`。
- 两个数据集都以单数据集、单阶段分类评估。种子为 0、1、42，分别抽取每类 1、5、10 张训练图片。按整个测试集计算总体准确率，然后对三个种子等权平均。标准差按三个配对提升计算。
- 1-shot 时，单张样本无法估计类内协方差，C 的实现会跳过共享；两个数据集上的全部逐图预测均相同。结论限于上述数据与种子。

## 复核与复现

每个数据集目录中的 `protocol.json` 保存样本 ID、类别提示词、权重哈希和参数，`runs.csv` 保存每个种子的准确率，`predictions.csv` 保存每张测试图的两组预测，`summary.csv` 保存三次平均值。报告中的准确率和配对差值已用逐图预测独立复算。

[Flowers102 逐次结果](flowers102/runs.csv) · [OrganAMNIST 逐次结果](organamnist/runs.csv) · [实验入口](../../run_paired_transfer.py)

在仓库根目录运行 `python scripts/prepare_flowers.py` 可生成 Flowers102 的 `records.json`。将该步骤下载的 `data/raw/oxford_flowers/102flowers.tgz`、医学数据和模型权重放入 `artifacts/paired_inputs/`。后两项分别下载自 [MedMNIST 的 OrganAMNIST-224](https://zenodo.org/records/10519652/files/organamnist_224.npz?download=1) 和 [OpenAI 的 CLIP ViT-B/16 权重](https://openaipublic.blob.core.windows.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)。

在支持 CUDA 的 PyTorch 环境中，使用 Python 3.11 或更高版本，安装仓库自带 CLIP 的依赖后运行：

```sh
python partC/run_paired_transfer.py flowers102
python partC/run_paired_transfer.py organamnist
```
