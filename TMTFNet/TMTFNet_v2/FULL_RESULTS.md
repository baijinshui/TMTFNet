# TMTFNet_v2 Complete Experimental Results

## Environment
- GPU: NVIDIA GeForce RTX 5090
- PyTorch version: 2.9.1+cu128
- CUDA version: 12.8
- Total experiment time: 99.22 minutes
- Protocol: fast

## Experiment 1: UCI HAR Classification

### Table 1: Main Results (mean ± std over 3 seeds)

| Model | Params | Accuracy (%) | F1-Macro (%) | F1-Weighted (%) | Precision (%) | Recall (%) |
|-------|--------|-------------|-------------|-----------------|---------------|------------|
| TMTFNet_v2 | 3557.0K | 93.72 ± 0.26 | 93.60 ± 0.29 | 93.70 ± 0.25 | 93.95 ± 0.38 | 93.70 ± 0.28 |
| LSTM | 967.0K | 92.69 ± 0.12 | 92.84 ± 0.12 | 92.66 ± 0.11 | 92.84 ± 0.16 | 92.88 ± 0.09 |
| GRU | 733.8K | 92.30 ± 0.41 | 92.36 ± 0.37 | 92.28 ± 0.41 | 92.46 ± 0.44 | 92.41 ± 0.33 |
| TCN | 564.5K | 91.51 ± 0.40 | 91.59 ± 0.40 | 91.49 ± 0.39 | 91.57 ± 0.43 | 91.72 ± 0.35 |
| Transformer | 679.4K | 90.32 ± 0.80 | 90.29 ± 0.82 | 90.32 ± 0.77 | 90.48 ± 0.70 | 90.41 ± 0.79 |
| Crossformer | 2068.0K | 90.26 ± 1.12 | 90.02 ± 1.19 | 90.21 ± 1.14 | 90.18 ± 1.23 | 90.06 ± 1.19 |
| PatchTST | 812.2K | 94.28 ± 0.58 | 94.33 ± 0.60 | 94.25 ± 0.58 | 94.56 ± 0.51 | 94.26 ± 0.63 |
| DLinear | 2.2K | 75.06 ± 2.94 | 74.16 ± 2.83 | 75.10 ± 2.77 | 74.60 ± 2.76 | 74.04 ± 2.94 |
| TimeMixer | 118.2K | 91.41 ± 0.39 | 91.41 ± 0.41 | 91.42 ± 0.41 | 91.56 ± 0.35 | 91.55 ± 0.42 |
| iTransformer | 628.9K | 88.59 ± 0.33 | 88.66 ± 0.34 | 88.58 ± 0.34 | 88.71 ± 0.39 | 88.68 ± 0.34 |

Best single-seed results per model:

| Model | Seed | Accuracy | F1-Macro | F1-Weighted | Train Time (s) |
|-------|------|----------|----------|-------------|----------------|
| TMTFNet_v2 | 456 | 93.96 | 93.88 | 93.94 | 103.74 |
| LSTM | 42 | 92.81 | 92.96 | 92.77 | 13.77 |
| GRU | 456 | 92.57 | 92.58 | 92.55 | 37.41 |
| TCN | 42 | 91.75 | 91.84 | 91.75 | 28.37 |
| Transformer | 456 | 91.01 | 90.98 | 90.97 | 9.47 |
| Crossformer | 42 | 90.97 | 90.77 | 90.95 | 57.55 |
| PatchTST | 456 | 94.74 | 94.82 | 94.72 | 39.48 |
| DLinear | 42 | 77.50 | 76.59 | 77.45 | 11.55 |
| TimeMixer | 123 | 91.72 | 91.73 | 91.73 | 42.22 |
| iTransformer | 42 | 88.87 | 88.94 | 88.87 | 17.04 |

## Experiment 2: ETTh1 Forecasting

### Table 2: ETTh1 Results (mean ± std over 2 seeds)

| Model | H=24 MSE | H=24 MAE | H=48 MSE | H=48 MAE | H=96 MSE | H=96 MAE |
|-------|---------|---------|---------|---------|---------|---------|
| TMTFNet_v2 | 0.0873 ± 0.0232 | 0.2300 ± 0.0369 | 0.1346 ± 0.0286 | 0.2841 ± 0.0344 | 0.1977 ± 0.0198 | 0.3534 ± 0.0186 |
| LSTM | 0.2347 ± 0.0175 | 0.3989 ± 0.0181 | 0.2334 ± 0.0319 | 0.3933 ± 0.0324 | 0.2789 ± 0.0568 | 0.4359 ± 0.0489 |
| GRU | 0.0759 ± 0.0039 | 0.2068 ± 0.0029 | 0.1895 ± 0.0578 | 0.3476 ± 0.0628 | 0.2593 ± 0.0225 | 0.4148 ± 0.0205 |
| TCN | 0.1419 ± 0.0284 | 0.2986 ± 0.0313 | 0.2152 ± 0.0103 | 0.3794 ± 0.0147 | 0.3612 ± 0.0085 | 0.4946 ± 0.0251 |
| Transformer | 0.0973 ± 0.0025 | 0.2384 ± 0.0041 | 0.1353 ± 0.0053 | 0.2842 ± 0.0073 | 0.2460 ± 0.0066 | 0.4031 ± 0.0097 |
| Crossformer | 0.0905 ± 0.0261 | 0.2313 ± 0.0411 | 0.1687 ± 0.0031 | 0.3281 ± 0.0024 | 0.2492 ± 0.0254 | 0.4064 ± 0.0210 |
| PatchTST | 0.0942 ± 0.0192 | 0.2339 ± 0.0299 | 0.1631 ± 0.0663 | 0.3186 ± 0.0779 | 0.2440 ± 0.0423 | 0.4061 ± 0.0429 |
| DLinear | 0.0475 ± 0.0014 | 0.1618 ± 0.0019 | 0.0746 ± 0.0036 | 0.2063 ± 0.0053 | 0.1072 ± 0.0020 | 0.2518 ± 0.0025 |
| TimeMixer | 0.2250 ± 0.0149 | 0.3915 ± 0.0146 | 0.2811 ± 0.0040 | 0.4391 ± 0.0020 | 0.2987 ± 0.0048 | 0.4514 ± 0.0045 |
| iTransformer | 0.4255 ± 0.0359 | 0.4991 ± 0.0275 | 0.6203 ± 0.2231 | 0.6489 ± 0.1657 | 0.7162 ± 0.0721 | 0.7365 ± 0.0439 |

## Experiment 3: Cross-Domain Forecasting

### Table 3: Cross-Domain Results (H=24, mean ± std over 2 seeds)

| Model | ETTh1→ETTh1 MSE | ETTh1→ETTh2 MSE | ETTh2→ETTh2 MSE | Transfer Gap |
|-------|----------------|----------------|----------------|-------------|
| TMTFNet_v2 | 0.0873 ± 0.0232 | 1.4079 ± 0.1633 | 0.3391 ± 0.0107 | 315.16% |
| LSTM | 0.2347 ± 0.0175 | 3.3437 ± 1.4994 | 0.3183 ± 0.0149 | 950.57% |
| GRU | 0.0759 ± 0.0039 | 2.1255 ± 0.4076 | 0.2388 ± 0.0019 | 790.14% |
| TCN | 0.1419 ± 0.0284 | 3.6227 ± 1.5295 | 0.3182 ± 0.0966 | 1038.69% |
| Transformer | 0.0973 ± 0.0025 | 2.7117 ± 0.3679 | 0.2234 ± 0.0179 | 1113.77% |
| Crossformer | 0.0905 ± 0.0261 | 1.8244 ± 0.3725 | 0.2159 ± 0.0029 | 744.97% |
| PatchTST | 0.0942 ± 0.0192 | 2.8547 ± 0.6871 | 0.1764 ± 0.0107 | 1518.45% |
| DLinear | 0.0475 ± 0.0014 | 0.3081 ± 0.0595 | 0.1276 ± 0.0016 | 141.45% |
| TimeMixer | 0.2250 ± 0.0149 | 3.0594 ± 0.0349 | 0.4248 ± 0.0267 | 620.24% |
| iTransformer | 0.4255 ± 0.0359 | 1.1140 ± 0.0696 | 0.5170 ± 0.0572 | 115.46% |

Transfer Gap = (ETTh1→ETTh2 MSE - ETTh2→ETTh2 MSE) / ETTh2→ETTh2 MSE × 100%

## Experiment 4: Ablation Study

### Table 4a: Ablation on UCI HAR (mean ± std over 2 seeds)

| Variant | Accuracy (%) | F1-Macro (%) | Δ Acc |
|---------|-------------|-------------|-------|
| TMTFNet_v2 (Full) | 93.59 ± 0.24 | 93.45 ± 0.26 | — |
| w/o G-CMTA | 92.79 ± 1.61 | 92.61 ± 1.69 | -0.80 |
| w/o AMG | 92.48 ± 0.07 | 92.30 ± 0.06 | -1.10 |
| w/o A-HTF | 93.03 ± 0.46 | 92.86 ± 0.51 | -0.56 |
| w/o AttnPool | 93.15 ± 0.19 | 92.99 ± 0.20 | -0.44 |
| w/o ModDrop | 93.16 ± 0.22 | 93.04 ± 0.27 | -0.42 |

### Table 4b: Ablation on ETTh1→ETTh2 Cross-Domain Forecasting (H=24)

| Variant | MSE | MAE | Δ MSE |
|---------|-----|-----|-------|
| TMTFNet_v2 (Full) | 1.4079 ± 0.1631 | 0.9807 ± 0.0451 | — |
| w/o G-CMTA | 1.3774 ± 0.0944 | 0.9791 ± 0.0206 | -0.0305 |
| w/o AMG | 1.2308 ± 0.1082 | 0.9195 ± 0.0449 | -0.1771 |
| w/o A-HTF | 1.5843 ± 0.1228 | 1.0515 ± 0.0351 | +0.1764 |
| w/o AttnPool | 1.4048 ± 0.1585 | 0.9793 ± 0.0434 | -0.0031 |
| w/o ModDrop | 2.4575 ± 0.8748 | 1.3041 ± 0.2650 | +1.0496 |

## Experiment 5: Hyperparameter Sensitivity

### Table 5a: Effect of d_model

| d_model | Params | Accuracy (%) | F1-Macro (%) |
|---------|--------|-------------|-------------|
| 32 | 265.6K | 93.28 | 93.14 |
| 64 | 947.0K | 94.06 | 93.94 |
| 128 | 3557.0K | 93.72 | 93.59 |
| 256 | 13765.9K | 93.48 | 93.42 |

### Table 5b: Effect of n_heads

| n_heads | Params | Accuracy (%) | F1-Macro (%) |
|---------|--------|-------------|-------------|
| 2 | 3557.0K | 92.77 | 92.60 |
| 4 | 3557.0K | 93.01 | 92.88 |
| 8 | 3557.0K | 93.72 | 93.59 |
| 16 | 3557.0K | 93.38 | 93.27 |

### Table 5c: Effect of n_layers

| n_enc_layers | Params | Accuracy (%) | F1-Macro (%) |
|---------|--------|-------------|-------------|
| 1 | 2367.4K | 93.32 | 93.21 |
| 2 | 2962.2K | 94.13 | 94.07 |
| 3 | 3557.0K | 93.72 | 93.59 |
| 4 | 4151.8K | 92.30 | 92.14 |

### Table 5d: Effect of modality_dropout

| modality_dropout | Params | Accuracy (%) | F1-Macro (%) |
|---------|--------|-------------|-------------|
| 0.0 | 3557.0K | 92.74 | 92.57 |
| 0.05 | 3557.0K | 93.28 | 93.17 |
| 0.1 | 3557.0K | 93.72 | 93.59 |
| 0.15 | 3557.0K | 93.96 | 93.88 |
| 0.2 | 3557.0K | 93.21 | 93.11 |

## Experiment 6: Efficiency Comparison

| Model | Params (K) | Train Time/epoch (s) | Inference Time/batch (ms) | Accuracy (%) |
|-------|-----------|---------------------|--------------------------|-------------|
| TMTFNet_v2 | 3557.0 | 1.85 | 13.82 | 93.76 |
| LSTM | 967.0 | 0.69 | 2.76 | 92.81 |
| GRU | 733.8 | 0.62 | 1.16 | 91.82 |
| TCN | 564.5 | 0.50 | 6.18 | 91.75 |
| Transformer | 679.4 | 0.53 | 0.75 | 90.50 |
| Crossformer | 2068.0 | 1.05 | 2.14 | 90.97 |
| PatchTST | 812.2 | 0.70 | 1.09 | 94.47 |
| DLinear | 2.2 | 0.33 | 0.61 | 77.50 |
| TimeMixer | 118.2 | 0.70 | 1.53 | 90.97 |
| iTransformer | 628.9 | 0.52 | 0.58 | 88.87 |

## Summary

- UCI HAR: TMTFNet_v2 achieves 93.72% accuracy, lower than the strongest baseline PatchTST by 0.55 percentage points.
- ETTh1 Forecasting: TMTFNet_v2 achieves 0.0873 MSE at H=24, worse than the strongest baseline DLinear.
- Cross-Domain: TMTFNet_v2 achieves 1.4079 MSE on ETTh1→ETTh2; best model under the evaluated protocol is DLinear with 0.3081 MSE.
- Ablation: on HAR, removing any module lowers accuracy (w/o G-CMTA, w/o AMG, w/o A-HTF, w/o AttnPool, w/o ModDrop). On ETTh1→ETTh2, the clearly helpful components are w/o A-HTF, w/o ModDrop, while w/o G-CMTA, w/o AMG, w/o AttnPool slightly reduce MSE when removed.
- Best configuration from sensitivity study: d_model=64, n_heads=8, n_layers=2, modality_dropout=0.15.
