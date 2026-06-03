# TMTFNet_v2 Supplementary Experimental Results

## Environment
- GPU: NVIDIA GeForce RTX 5090
- Total time: 35.20 minutes

## Experiment A: Hyperparameter Sensitivity on ETTh1 Forecasting (H=24, seed=42)

### A1: Effect of d_model
| d_model | Params (K) | MSE | MAE |
|---------|-----------|-----|-----|
| 32 | 152.9 | 0.0979 | 0.2435 |
| 64 | 531.0 | 0.1393 | 0.2978 |
| 128 | 1963.1 | 0.0709 | 0.2039 |
| 256 | 7530.6 | 0.0651 | 0.1936 |

### A2: Effect of n_heads
| n_heads | MSE | MAE |
|---------|-----|-----|
| 2 | 0.0864 | 0.2207 |
| 4 | 0.1024 | 0.2441 |
| 8 | 0.0709 | 0.2039 |
| 16 | 0.0792 | 0.2174 |

### A3: Effect of n_enc_layers
| n_enc_layers | Params (K) | MSE | MAE |
|-------------|-----------|-----|-----|
| 1 | 1170.0 | 0.0649 | 0.1948 |
| 2 | 1566.5 | 0.0794 | 0.2178 |
| 3 | 1963.1 | 0.0709 | 0.2039 |
| 4 | 2359.6 | 0.0676 | 0.2017 |

### A4: Effect of modality_dropout
| modality_dropout | MSE | MAE |
|-----------------|-----|-----|
| 0.00 | 0.0835 | 0.2189 |
| 0.05 | 0.0730 | 0.2076 |
| 0.10 | 0.0709 | 0.2039 |
| 0.15 | 0.1191 | 0.2697 |
| 0.20 | 0.0752 | 0.2112 |

## Experiment B: Hyperparameter Sensitivity on ETTh1→ETTh2 Cross-Domain (H=24, seed=42)

### B1: Effect of d_model
| d_model | Cross-Domain MSE |
|---------|-----------------|
| 32 | 1.7251 |
| 64 | 1.0711 |
| 128 | 1.5233 |
| 256 | 1.5374 |

### B2: Effect of n_heads
| n_heads | Cross-Domain MSE |
|---------|-----------------|
| 2 | 1.1585 |
| 4 | 1.4111 |
| 8 | 1.5233 |
| 16 | 1.1864 |

### B3: Effect of n_enc_layers
| n_enc_layers | Cross-Domain MSE |
|-------------|-----------------|
| 1 | 1.2846 |
| 2 | 1.3303 |
| 3 | 1.5233 |
| 4 | 1.6016 |

### B4: Effect of modality_dropout
| modality_dropout | Cross-Domain MSE |
|-----------------|-----------------|
| 0.00 | 1.8390 |
| 0.05 | 1.5458 |
| 0.10 | 1.5233 |
| 0.15 | 1.4026 |
| 0.20 | 1.5171 |

## Experiment C: ETTm1 Forecasting (optional, seed=42)

### ETTm1 Results
| Model | H=24 MSE | H=24 MAE | H=48 MSE | H=48 MAE | H=96 MSE | H=96 MAE |
|-------|---------|---------|---------|---------|---------|---------|
| TMTFNet_v2 | 0.0287 | 0.1260 | 0.0412 | 0.1544 | 0.0638 | 0.1898 |
| LSTM | 0.0353 | 0.1408 | 0.0506 | 0.1706 | 0.1221 | 0.2709 |
| GRU | 0.0206 | 0.1037 | 0.0590 | 0.1817 | 0.0792 | 0.2156 |
| TCN | 0.0369 | 0.1481 | 0.0488 | 0.1645 | 0.1303 | 0.2901 |
| Transformer | 0.0265 | 0.1244 | 0.0436 | 0.1673 | 0.0764 | 0.2080 |
| Crossformer | 0.0224 | 0.1106 | 0.0442 | 0.1596 | 0.0961 | 0.2399 |
| PatchTST | 0.0304 | 0.1296 | 0.0396 | 0.1490 | 0.0705 | 0.1986 |
| DLinear | 0.0171 | 0.0904 | 0.0301 | 0.1216 | 0.0452 | 0.1517 |
| TimeMixer | 0.0636 | 0.1973 | 0.0894 | 0.2363 | 0.1356 | 0.2977 |
| iTransformer | 0.3385 | 0.3759 | 0.4016 | 0.4304 | 0.4513 | 0.4893 |
