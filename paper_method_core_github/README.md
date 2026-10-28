# 面向弱监督视频异常检测的 Short–Long SDF 方法

本文件夹包含 Short/Long SDF 方法在 XD-Violence 数据集上的 CLIP 10-crop 核心实现。文件夹中仅提供源代码和运行说明，不包含数据集、提取后的特征、模型检查点、日志或论文 PDF。

## 文件夹内容

- `code/main_short_long_sdf_10crop_try.py`：训练入口程序。
- `code/context_score_short_long_sdf_10crop_try.py` 及其模型和时序混合模块依赖：实现短分支、长分支、SDF 证据计算和分数融合。
- `code/train_context_score_try.py` 和 `code/train_semantic_mag_try.py`：训练损失函数。
- `code/dataset.py`、`code/utils.py` 和 `code/test_component_auc_10crop_try.py`：数据加载、预处理和评估。
- `code/setup_xd_lists.py`、`code/merge_xd_10crop_features.py` 和 `code/check_xd_data.py`：数据列表生成、特征准备和数据检查工具。

## 环境依赖

请使用 Python 3.10 或更高版本。首先根据 CPU/CUDA 环境，从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择并安装合适的 PyTorch 版本。然后安装其他 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

## 数据准备

训练代码需要 XD-Violence 的 CLIP 10-crop 特征，特征形状为 `[T, 10, 512]`。请从获得授权的来源获取数据集和特征；不要将特征数组或受许可限制的数据内容提交到本仓库。

在本文件夹目录下运行以下命令，生成 XD 训练/测试列表和真实标注文件：

```bash
python code/setup_xd_lists.py
```

生成的数据列表会引用以下目录中的特征文件：

```text
features/xd_clip_10crop/train/
features/xd_clip_10crop/test/
```

请将相应特征文件放入这些目录，并确保文件名与列表中的记录一致。训练数据集实现默认认为训练列表的前 1,905 条为异常样本，其余为正常样本；训练前请确认列表顺序符合此假设。运行以下命令可检查数据和真实标注（GT）的对齐情况：

```bash
python code/check_xd_data.py
```

如果现有特征是按 crop 分别保存的，可以使用 `code/merge_xd_10crop_features.py` 将它们合并。请通过 `--src-train` 和 `--src-test` 参数指定符合本地特征目录结构的路径。

## 训练 XD 基线模型

请在本文件夹目录下运行：

```bash
bash run_xd.sh
```

该脚本使用原项目中的 XD 可复现实验配置：

```bash
python code/main_short_long_sdf_10crop_try.py \
  --feature_extractor clip \
  --dataset xd \
  --gpus 0 \
  --max-epoch 45 \
  --early-stop-drop 0.015 \
  --early-stop-min-epoch 25 \
  --seed 237033 \
  --context-mode sdf \
  --component-lam 0.05 \
  --context-gap-lam 0.05 \
  --center-lam 0.0 \
  --sparse-lam 0.01 \
  --scale-gate-mode learned \
  --sdf-prior-mode mean \
  --num-normal-centers 4 \
  --sdf-topk 3 \
  --short-sdf-weight 0.2 \
  --long-sdf-weight 0.2 \
  --short-long-fusion learned \
  --long-head mlp \
  --long-temporal-kernel 3 \
  --train-score-smooth-mode max \
  --train-score-smooth-kernel 5 \
  --test-smooth-mode max \
  --test-smooth-kernel 5 \
  --deterministic
```

请将 `--gpus` 设置为当前机器上可见的 GPU 编号。模型检查点和运行配置 JSON 文件会保存到 `result/` 目录。

## 许可证与引用

本文件夹目前未声明许可证。公开发布前，请确认你有权再分发这些实现，并添加适当的软件许可证和论文引用。数据集、特征和标注的许可与源代码许可相互独立。
