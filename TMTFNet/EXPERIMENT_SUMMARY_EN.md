# Experimental Summary for TMTFNet

## 1. Overall Assessment

The current experimental results present a mixed but still publishable picture of TMTFNet. On real-data benchmarks, TMTFNet is competitive rather than uniformly dominant. In activity recognition on UCI HAR, it achieves strong performance but does not outperform the best baseline. In standard in-domain forecasting on ETTh1, Transformer-based baselines remain stronger across all tested prediction horizons. However, TMTFNet shows its clearest advantage in cross-domain forecasting, where it yields the lowest MSE on the ETTh1 to ETTh2 transfer setting. Therefore, the most defensible paper narrative is not that TMTFNet is universally superior, but that it provides a favorable trade-off between multi-modal temporal modeling and transfer robustness, with particular value in cross-domain forecasting scenarios.

## 2. Main Results on Real Datasets

### UCI HAR Classification

On the UCI HAR dataset, TMTFNet achieved an accuracy of 90.02% and a macro-F1 score of 89.75%. Although this confirms that the model is effective for multi-modal activity recognition, it did not outperform all baselines. The best result was obtained by TCN with 92.37% accuracy and 92.46% macro-F1, followed by GRU with 90.97% accuracy and 91.07% macro-F1. TMTFNet still outperformed LSTM, Transformer, and Crossformer, indicating that it remains a competitive solution for sensor-based classification, but not the strongest architecture under the current configuration.

### ETTh1 Forecasting

On the ETTh1 forecasting benchmark, TMTFNet did not achieve the best in-domain accuracy. For the 24-step setting, TMTFNet obtained an MSE of 0.3234 and an MAE of 0.4420, while Transformer achieved the best result with an MSE of 0.2954 and an MAE of 0.4177. A similar pattern was observed for longer horizons. At 48 steps, TMTFNet reached an MSE of 0.4997 compared with 0.4858 for Transformer. At 96 steps, the gap became more evident, with TMTFNet obtaining an MSE of 0.7038 while Transformer achieved 0.6329. These results suggest that the proposed architecture does not currently surpass strong Transformer baselines in standard single-domain long-horizon forecasting.

### Cross-Domain Forecasting

The strongest evidence in favor of TMTFNet appears in the cross-domain forecasting experiment. In the in-domain ETTh1 to ETTh1 setting, Transformer remained the best model with an MSE of 0.2954, compared with 0.3234 for TMTFNet. However, in the cross-domain ETTh1 to ETTh2 setting, TMTFNet achieved the lowest MSE of 0.2899, outperforming Transformer (0.3367) and LSTM (0.3051). This corresponds to an MSE reduction of approximately 13.9% relative to Transformer and 5.0% relative to LSTM. These findings support the claim that TMTFNet is particularly beneficial when domain shift is present.

## 3. Ablation Findings

The ablation study does not fully support the assumption that all proposed modules contribute positively under the current implementation. The full TMTFNet obtained 90.30% accuracy, while removing CMTA led to a minor decrease to 89.99%, and removing AMG reduced performance more noticeably to 89.24%. In contrast, removing HTF improved accuracy to 91.96%, which is higher than the complete model. This indicates that AMG provides the clearest positive contribution, CMTA has only a marginal effect, and HTF may currently introduce optimization difficulty or unnecessary complexity for this classification task. As a result, the paper should avoid claiming that every component is consistently beneficial across benchmarks.

## 4. Cross-Domain Classification and Auxiliary Experiments

The synthetic cross-domain classification experiment also yields mixed evidence. In the near-shift setting, TMTFNet reached 100% accuracy and matched LSTM while outperforming Transformer. However, under the far-shift setting, TMTFNet dropped sharply to 22.42%, compared with 84.33% for LSTM. This means the proposed method does not yet demonstrate stable superiority in synthetic domain-transfer classification.

The scalability experiment and the multi-seed synthetic significance test are not very informative in their current form. In the scalability study, all tested TMTFNet variants achieved 100% accuracy, and in the five-seed synthetic experiment every model also achieved 100% mean accuracy with zero standard deviation. These results indicate that the synthetic task is saturated and therefore cannot serve as strong evidence for model discrimination or statistical significance.

## 5. Recommended Paper Narrative

Based on the current results, the most defensible manuscript framing is as follows:

TMTFNet is a competitive multi-modal temporal model whose main advantage lies in cross-domain forecasting rather than universal dominance on all in-domain benchmarks. On UCI HAR, it delivers strong performance but is outperformed by TCN and GRU. On ETTh1 forecasting, it remains competitive yet trails a standard Transformer baseline across all prediction horizons. Nevertheless, TMTFNet achieves the best performance in cross-domain forecasting from ETTh1 to ETTh2, suggesting that its fusion and alignment design improves robustness under distribution shift. Ablation results further show that the adaptive modality gating mechanism contributes positively, whereas the hierarchical temporal fusion block may require refinement.

## 6. Manuscript-Ready Summary Paragraph

The experimental results show that TMTFNet is an effective but not uniformly dominant architecture for cross-domain sequence modeling. On UCI HAR, TMTFNet achieved 90.02% accuracy and 89.75% macro-F1, demonstrating strong multi-modal classification capability, although TCN and GRU obtained higher final scores. On ETTh1 forecasting, TMTFNet remained competitive but did not surpass the Transformer baseline for 24-, 48-, or 96-step prediction. In contrast, TMTFNet showed a clear advantage in cross-domain forecasting, where it achieved the lowest MSE of 0.2899 on the ETTh1 to ETTh2 transfer task, outperforming both Transformer and LSTM. Ablation results indicate that adaptive modality gating is useful, while the hierarchical temporal fusion component may require further refinement. Overall, the results suggest that the main strength of TMTFNet lies in improving robustness under domain shift rather than consistently maximizing in-domain benchmark accuracy.

## 7. Quantitative Snapshot

| Experiment | Setting | Best Model | TMTFNet | Key Interpretation |
|---|---|---:|---:|---|
| Exp1 | UCI HAR classification | TCN: 92.37% Acc | 90.02% Acc | Competitive, but not best |
| Exp2 | ETTh1, pred=24 | Transformer: 0.2954 MSE | 0.3234 MSE | TMTFNet trails Transformer |
| Exp2 | ETTh1, pred=48 | Transformer: 0.4858 MSE | 0.4997 MSE | Small gap |
| Exp2 | ETTh1, pred=96 | Transformer: 0.6329 MSE | 0.7038 MSE | Larger gap at long horizon |
| Exp3 | Ablation on HAR | NoHTF: 91.96% Acc | 90.30% Acc | HTF needs refinement |
| Exp4 | Far synthetic transfer | LSTM: 84.33% Acc | 22.42% Acc | Weak evidence for classification transfer |
| Exp6 | ETTh1 to ETTh2 | TMTFNet: 0.2899 MSE | 0.2899 MSE | Strongest positive result |

## 8. Writing Guidance

If this summary is used for paper drafting, the Results and Discussion sections should emphasize three points: first, TMTFNet is competitive on real multi-modal classification; second, its strongest benefit appears in cross-domain forecasting; third, some proposed modules, especially HTF, still need refinement. A paper written around these claims will be substantially more credible than one claiming consistent superiority across all tasks.
