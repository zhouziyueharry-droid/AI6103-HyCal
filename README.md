# AI6103 · HyCal 少样本分类

每类只有五张标注图时，能否通过共享各类内部的变化来改善分类？本仓库整合三个小组成员的 HyCal 实现、数据准备脚本和实验记录。所有方法使用冻结的 CLIP。团队接下来的对照是在 Flowers102 和 OrganAMNIST 上运行同一份 C 方法，固定图像、特征和评分参数，只改变协方差共享比例。

## 从哪里开始

| 工作 | 代码 | 可复核的材料 |
|---|---|---|
| A · 花卉、纹理、飞机 | [算法](src/)、[数据准备与实验入口](scripts/) | [固定抽样](data/manifests/)、[预测](results/part_a_round2/)、[统计表](reports/part_a_round2/analysis/) |
| B · 医学 CT | [OrganAMNIST 加载器](external/HyCal-CIL/task_datasets/organamnist.py)、[集群环境检查](external/HyCal-CIL/scripts/tc2_gpu_check.sh) | B 的交付包只提供了官方代码和环境检查，医学门控的最终实现及逐种子预测尚待成员补充。 |
| C · 卫星和星系 | [协方差估计与打分](partC/hycal_cached.py)、[固定参数消融](partC/ablation_shrink.py) | [逐项消融表](partC/results/param_search/ablation_shrink.md)、[验证集选出的参数](partC/results/param_search/selected.json)、[固定抽样](partC/indices/) |

### 先验证历史结果

以下命令从仓库根目录运行。第一条使用 Python 3.11+ 标准库，从已保存的预测复算 A 的 20 组指标，无需显卡、模型权重或原图。第二条运行该复算工具的自测。

```sh
python tools/handoff/verify_predictions.py
python -m unittest discover -s tools/handoff -p test_handoff.py
```

### C 的方法怎么跑

已有兼容的 PyTorch、torchvision 和 NumPy 环境可以直接复用。[原 macOS 环境记录](partC/requirements-macos.txt)列有其余依赖；重新编码需要 Pillow、ftfy、regex、tqdm 和 OpenAI CLIP ViT-B/16 权重。原版 HyCal 的依赖另见[上游目录](external/HyCal-CIL/requirements.txt)。

现成的 EuroSAT / Galaxy10 特征缓存放在 `partC/cache/{dataset}_ViT-B-16.npz`。缓存包含各划分的特征、标签、路径和类别文本特征，字段见[提取脚本](partC/extract_features.py)。有两套缓存和仓库内的固定索引后，可从根目录运行：

```sh
python partC/ablation_shrink.py
```

脚本会重写 [C 的消融结果表](partC/results/param_search/ablation_shrink.md)。表中的 **D 减 B** 比较共享比例 0.75 和 0，使用相同 support、同一组打分权重与岭正则。EuroSAT 5-shot 的历史记录为 76.60% → 79.53%。此处的 B 和 D 是两组配置名。C 的原始数据加载入口目前支持 EuroSAT 与 Galaxy10；Flowers102 和 OrganAMNIST 的统一对照属于下一阶段。

## 数据与复现范围

- A 的 [Flowers102 准备脚本](scripts/prepare_flowers.py)生成当前机器的样本路径和来源记录。公开仓保留可迁移的类别映射与 support 索引。历史锁针对当时的完整工作区，另行复算全部旧锁需要对应版本的特征缓存与原始资产。
- C 的 EuroSAT RGB 使用 CoOp/Zhou 划分；Galaxy10 DECaLS 使用原先的分层划分。[数据目录约定](partC/common.py)和[索引生成脚本](partC/make_support_indices.py)定义输入格式。
- 官方 HyCal 和随附的 CLIP 统一位于 [external/HyCal-CIL](external/HyCal-CIL/)，源自[上游固定提交](https://github.com/EJLEE5826/HyCal-CIL/tree/4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41)。B 包内的 TC2 检查脚本也保存在该目录。上游 [LICENSE](external/HyCal-CIL/LICENSE) 和 [CLIP 许可说明](external/HyCal-CIL/THIRD_PARTY_NOTICES.md)随代码保留。若运行 A 的严格上游对照 `scripts/verify_official.py`，需将 `HYCAL_OFFICIAL_CHECKOUT` 指向该提交的干净独立 Git checkout，供脚本校验真实提交身份。
- 原图、模型权重和特征缓存由各成员单独准备；仓内可共享配置、索引、方法代码与结果。新实验请保留数据集、种子、配置及逐种子结果，方便组员复核。
