# TMTFNet: Transformer-Based Multi-Modal Temporal Fusion Network

## Paper: "TMTFNet: Transformer-Based Multi-Modal Temporal Fusion Network with Cross-Modal Attention and Domain-Adaptive Alignment for Cross-Domain Sequence Modeling"

---

## 🏗️ Architecture Overview

TMTFNet consists of 5 key components:

1. **Modality-Specific Temporal Encoder (MSTE)**: Lightweight transformer encoder per modality with learnable + sinusoidal positional encoding
2. **Cross-Modal Temporal Attention (CMTA)**: Bidirectional cross-attention between modalities at each temporal position *(Key Innovation)*
3. **Adaptive Modality Gating (AMG)**: Dynamic gating mechanism that weights modality contributions based on temporal context *(Key Innovation)*
4. **Hierarchical Temporal Fusion (HTF)**: Multi-scale convolution + scale-wise attention for capturing temporal patterns at different granularities *(Key Innovation)*
5. **Domain-Adaptive Contrastive Alignment (DACA)**: Gradient reversal + domain classifier for cross-domain transfer

## 📊 Experiments

| # | Experiment | Task | Dataset | Purpose |
|---|-----------|------|---------|---------|
| 1 | Multi-Modal Classification | Classification | UCI HAR / Synthetic | Main result: multi-modal fusion effectiveness |
| 2 | Time Series Forecasting | Forecasting | ETTh1 / Synthetic | Multi-horizon forecasting comparison |
| 3 | Ablation Study | Classification | Same as Exp1 | Component contribution analysis |
| 4 | Cross-Domain Transfer | Classification | Synthetic (3 domains) | Domain adaptation capability |
| 5 | Scalability Analysis | Classification | Synthetic | Parameter efficiency study |
| 6 | Cross-Domain Forecasting | Forecasting | ETTh1→ETTh2 / Synthetic | Cross-domain transfer in forecasting |
| * | Multi-Seed Testing | Classification | Synthetic | Statistical significance (5 seeds) |

## 📦 Datasets

### Download Links

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| UCI HAR | [UCI ML Repo](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip) | ~60MB | 6 activities, 9 sensor channels (acc+gyro), 10299 samples |
| ETTh1/h2 | [GitHub](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) | ~2MB each | Electricity transformer temp, hourly, 7 features |
| ETTm1/m2 | [GitHub](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) | ~8MB each | Same, 15-min granularity |

### Direct Download URLs
```
# UCI HAR
https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

# ETT (individual CSV files)
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv
```

### Automatic Download
```bash
chmod +x download_data.sh
./download_data.sh
```

### Expected Directory Structure After Download
```
data/
├── UCI HAR Dataset/
│   └── UCI HAR Dataset/
│       ├── train/
│       │   ├── Inertial Signals/
│       │   │   ├── body_acc_x_train.txt
│       │   │   ├── body_acc_y_train.txt
│       │   │   └── ...
│       │   └── y_train.txt
│       └── test/
│           ├── Inertial Signals/
│           └── y_test.txt
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
└── ETTm2.csv
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda activate boluo_huo
pip install -r requirements.txt
```

### 2. Run with Synthetic Data (fast test, ~15 min on RTX 5090)
```bash
python run_experiments.py --exp all
```

### 3. Run with Real Data (full paper experiments)
```bash
# Download data first
bash download_data.sh

# Run all experiments
python run_experiments.py --use_real_data --exp all
```

### 4. Run Individual Experiments
```bash
python run_experiments.py --exp exp1    # Classification
python run_experiments.py --exp exp2    # Forecasting
python run_experiments.py --exp exp3    # Ablation
python run_experiments.py --exp exp4    # Cross-domain
python run_experiments.py --exp exp5    # Scalability
python run_experiments.py --exp exp6    # Cross-domain forecast
python run_experiments.py --exp multi_seed  # Statistical significance
```

### 5. Or use the all-in-one script
```bash
chmod +x run_all.sh
./run_all.sh           # synthetic
./run_all.sh --real    # real data
```

## ⚙️ Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 64 | Hidden dimension |
| n_heads | 4 | Attention heads |
| n_layers | 2 | Encoder layers per modality |
| batch_size | 128 | Batch size |
| max_epochs | 50 | Maximum epochs |
| patience | 10 | Early stopping patience |
| lr | 0.001 | Learning rate |
| dropout | 0.1 | Dropout rate |

## 📁 Output Structure
```
results/
├── exp1_classification/
│   ├── results.json          # All metrics
│   └── *.pt                  # Model checkpoints
├── exp2_forecast_*/
├── exp3_ablation/
├── exp4_cross_domain/
├── exp5_scalability/
├── exp6_cross_forecast/
└── multi_seed/

figures/
├── exp1/
│   ├── training_curves.png
│   ├── performance_comparison.png
│   ├── tsne.png
│   ├── confusion_matrix.png
│   ├── ablation.png
│   └── param_efficiency.png
├── exp2_*/
├── exp3/
├── exp4/
├── exp5/
└── exp6/
```

## 📄 After Running: Paper Writing

Once experiments are complete, share the following with Claude for paper writing:
1. All `results.json` files from `./results/`
2. All figures from `./figures/`
3. Any specific observations or notes

Claude will then write a complete SCI-level paper including:
- Abstract, Introduction, Related Work
- Methodology (with architecture diagram description)
- Experimental Setup
- Results & Analysis
- Ablation Study
- Conclusion
