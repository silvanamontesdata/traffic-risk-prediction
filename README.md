# Machine Learning for Human Behavior: A Road Safety Case Study

**How much does model complexity add when predicting self-reported driving behavior?** This case study applies supervised machine learning to two outcomes in road-safety research. It compares familiar regression baselines with regularized, tree-based and neural-network models under shared cross-validation splits.

![Mean cross-validated model performance in the two road-safety tasks](figures/model-comparison.png)

## Data and measures

The analysis uses a restricted road-safety research dataset with 792 participant records. Personality was measured with the Personality Inventory for DSM-5 (PID-5), originally developed by Krueger et al. (2012). Sánchez, Montes, and colleagues (2023) later reported on a shortened and modified version for use in Argentina. The continuous outcome sums 12 self-reported driving-risk behaviors rated from 1 to 5, for a theoretical range of 12–60. It was observed for 777 participants. The classification outcome is self-reported crash involvement as coded in the source study (135 positives among 792). The classification predictors are five personality domains, five driving-style scores (stress reduction excluded) and five specific behaviors (R1–R5): speeding, driving after drinking, distracted driving, phone use and driving while tired. The regression predictors are either 25 personality facets or five domains.

## What did the models learn?

| Task | Model | Mean validation performance |
|---|---|---:|
| Risk-behavior score, 25 facets | Linear regression | R² .237 · MAE 4.86 |
| Risk-behavior score, 25 facets | Elastic Net | R² .251 · MAE 4.82 |
| Risk-behavior score, 25 facets | Random Forest | R² .255 · MAE 4.76 |
| Recorded crash involvement | Logistic regression | ROC-AUC .616 · AP .268 |
| Recorded crash involvement | Decision tree | ROC-AUC .568 · AP .205 |
| Recorded crash involvement | Random Forest | ROC-AUC .591 · AP .240 |
| Recorded crash involvement | Multilayer perceptron | ROC-AUC .579 · AP .235 |

The forest reduced error by about **0.10 score points** relative to ordinary linear regression with 25 facets. The mean-only reference had MAE 5.71. Crash involvement was harder to discriminate. Logistic regression had the highest observed mean ROC-AUC (.616) and average precision (.268). The prevalence was .170, a reference for AP. These are descriptive differences from internal validation in a previously explored sample. The crash outcome is retrospective and the analysis does not evaluate prediction of future crashes.

[Decision tree](figures/decision-tree.png) · [Correlation heatmap](figures/correlation-heatmap.png)

For the illustrated tree, a stratified 70/30 split assigned 554 participants to training and 238 to testing. Imputation and tree fitting used the training data. The tree starts with disinhibition, followed by dissociative and risky driving styles. Its ROC-AUC on the held-out test was .566. The table reports mean performance from five-fold cross-validation, which assesses each model across several train/test splits. The heatmap shows descriptive Pearson correlations among five driving-style scores and five PID-5 domains.

## Evaluation and reproducibility

Each task uses five shared folds, stratified for classification. Mean imputation and scaling, where applicable, are fitted inside the training folds. Model settings were fixed. The classification output also includes Brier score and balanced accuracy at a threshold of 0.5. Fold standard deviations describe variability across folds.

    python -m pip install -r requirements.txt
    export TRANSITO_CSV_PATH=/path/outside/this/project/authorized.csv
    python analysis.py

The participant data stay in restricted access. The project includes code and aggregate results only. The script regenerates the figures and all tables.

[Classification metrics](results/classification-summary.csv) · [70/30 tree test](results/tree-holdout.csv) · [Regression metrics](results/regression-summary.csv)

## References

Krueger, R. F., Derringer, J., Markon, K. E., Watson, D., & Skodol, A. E. (2012). Initial construction of a maladaptive personality trait model and inventory for DSM-5. *Psychological Medicine, 42*(9), 1879–1890. https://doi.org/10.1017/S0033291711002674

Sánchez, R. O., Montes, S. A., Galarza, A. L., Somerstein, L. D., & Gainza, M. D. (2023). Una versión reducida y modificada del Inventario de Personalidad para el DSM-5 (PID-5) en la Argentina. *Interdisciplinaria, 40*(1), 115–136. https://doi.org/10.16888/interd.2023.40.1.7
