# TMTFNet — 跨域序列建模实验代码

本压缩包**只包含跑实验所需的代码**（模型、数据加载、训练器、各实验脚本、绘图/出表脚本）。
**不包含**数据集、实验输出（results/figures）、检查点、数据下载脚本和任何凭据。

## 目录结构

```
.
├── README_CODE_PACKAGE.md        # 本文件
├── EXPERIMENT_SUMMARY_EN.md      # 实验总览（原始版本）
├── requirements.txt              # 顶层依赖（转引 TMTFNet/requirements.txt）
├── run_all.sh / run_experiments.py  # 顶层入口（转发到 TMTFNet/）
│
├── TMTFNet/                      # 原始投稿版本代码 (v1)
│   ├── src/                      # models.py / datasets.py / trainer.py / visualize.py
│   ├── run_experiments.py        # v1 实验主入口
│   ├── run_all.sh
│   └── requirements.txt
│
└── TMTFNet_v2/                   # 当前/修订版本代码 (v2，rebuttal 用，建议以此为准)
    ├── src/
    │   ├── models.py             # TMTFNet 及各变体
    │   ├── datasets.py / datasets_fast.py
    │   ├── trainer.py / trainer_fast.py / trainer_da.py
    │   ├── alignment.py          # 跨域对齐 (CORAL / 一致性 / DANN 等)
    │   └── baselines_classical.py / baselines_cnn.py / baselines_da.py
    ├── run_rebuttal_experiments.py   # ★ 统一的 5-seed 实验主入口
    ├── launch_rebuttal.sh        # GPU 排队 + 断点续跑的自动化启动器
    ├── run_*.py                  # 各类专项 / 消融 / 蒸馏 / 敏感性实验脚本
    ├── make_figures.py / plot_*.py   # 图表生成
    ├── make_latex_numbers.py     # 导出论文用的 LaTeX 数字
    └── *_RESULTS.md              # 实验说明 / 结果汇总文档
```

> 说明：`TMTFNet_v2/` 是最近一次修订（rebuttal）所用的最新代码，论文中的结果以此为准；
> `TMTFNet/` 为原始投稿版本，保留以便对照。

## 运行环境

- Python ≥ 3.10，单卡 NVIDIA GPU（开发时用 RTX 5090）
- Conda 环境名：`zhangyue`
- 依赖：`torch>=2.0`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`、`pyyaml`、`tqdm`

```bash
conda activate zhangyue          # 或新建环境后安装依赖
pip install -r TMTFNet/requirements.txt
```

## 数据（需自行准备，本包不含）

代码默认从仓库根目录的 `data/` 读取数据，即把数据放成：

```
data/
├── UCI HAR Dataset/             # UCI HAR
├── PAMAP2_Dataset/Protocol/     # PAMAP2 (12 类)
├── ETTh1.csv  ETTh2.csv         # ETT 时序预测
└── ETTm1.csv  ETTm2.csv
```

公开数据来源（仅作参考，自行下载）：

- UCI HAR：`https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip`
- PAMAP2：`https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip`
- ETT (ETTh1/h2/m1/m2)：`https://github.com/zhouhaoyi/ETDataset` 的 `ETT-small/` 目录

## 如何跑实验（以 v2 为准）

主入口 `TMTFNet_v2/run_rebuttal_experiments.py`，默认 5 个随机种子 `(42, 123, 456, 789, 2024)`，
每个 `(seed, model, experiment)` 组合结果缓存为 JSON，重复运行只补算缺失的组合（可断点续跑）。

```bash
cd TMTFNet_v2

# 跑单个实验
python run_rebuttal_experiments.py --only exp1a_har_uci

# 指定种子
python run_rebuttal_experiments.py --only exp2_forecast --seeds 42 123 456

# 跑全部实验
python run_rebuttal_experiments.py

# 只看将运行哪些组合，不实际训练
python run_rebuttal_experiments.py --dry-run
```

可用的实验 key：

| key | 内容 |
|-----|------|
| `exp1a_har_uci`       | UCI HAR：深度基线 + TMTFNet 变体 + 域自适应模型 |
| `exp1b_har_classical` | UCI HAR：SVM / RF / kNN / LR 经典基线 |
| `exp1c_har_pamap2`    | PAMAP2（12 类）受试者内 + 跨受试者 HAR |
| `exp2_forecast`       | ETTh1 / ETTh2 / ETTm1 预测，H ∈ {24, 48, 96} |
| `exp3_cross_forecast` | 跨域预测 ETTh1↔ETTh2、ETTm1↔ETTm2、ETTh1→ETTm1 |
| `exp3_da_forecast`    | 上述跨域 + DANN / CoDATS / AdvSKM / RAINCOAT 等 DA 方法 |
| `exp4_ablation`       | TMTFNet 在 HAR / 跨域 / 对齐模块上的消融 |
| `exp5_sensitivity`    | 超参敏感性扫描（5 seeds） |

自动化（自带 GPU 显存排队 + 续跑）：

```bash
cd TMTFNet_v2
bash launch_rebuttal.sh          # 注意：脚本里硬编码了本机 python 路径，换机请改 PY=...
```

出图 / 出表（实验跑完、生成 `results/` 后）：

```bash
cd TMTFNet_v2
python make_figures.py           # 生成论文图
python make_latex_numbers.py     # 导出 LaTeX 数字
python measure_inference_latency.py   # 推理延迟测量
```

## 跑 v1（原始版本）

```bash
# 仓库根目录
python run_experiments.py --use_real_data --exp all
# 或
bash run_all.sh --real
```

> 输出会写到各自目录下的 `results/` 与 `figures/`（本包已剔除，运行后自动重建）。
