# Machine Learning for Human Behavior: A Road Safety Case Study

**How much does model complexity add when predicting self-reported driving behavior?** This case study applies supervised machine learning to two outcomes in road-safety research. It compares familiar regression baselines with regularized, tree-based and neural-network models under shared cross-validation splits.

![Mean cross-validated model performance in the two road-safety tasks](figures/model-comparison.png)

## Data and measures

The analysis uses a restricted road-safety research dataset with 792 participant records. Personality was measured with the Personality Inventory for DSM-5 (PID-5). The continuous outcome sums 12 self-reported driving-risk behaviors rated from 1 to 5, for a theoretical range of 12–60. It was observed for 777 participants. The classification outcome is self-reported crash involvement as coded in the source study (135 positives among 792). The classification predictors are five personality domains, five driving-style scores (stress reduction excluded) and the 12 behavior items. The regression predictors are either 25 personality facets or five domains.

## What did the models learn?

| Task | Model | Mean validation performance |
|---|---|---:|
| Risk-behavior score, 25 facets | Linear regression | R² .237 · MAE 4.86 |
| Risk-behavior score, 25 facets | Elastic Net | R² .251 · MAE 4.82 |
| Risk-behavior score, 25 facets | Random Forest | R² .255 · MAE 4.76 |
| Recorded crash involvement | Logistic regression | ROC-AUC .594 · AP .250 |
| Recorded crash involvement | Decision tree | ROC-AUC .580 · AP .214 |
| Recorded crash involvement | Random Forest | ROC-AUC .601 · AP .256 |
| Recorded crash involvement | Multilayer perceptron | ROC-AUC .554 · AP .245 |

The forest reduced error by about **0.10 score points** relative to ordinary linear regression with 25 facets. The mean-only reference had MAE 5.71. Crash involvement was harder to discriminate. The forest and logistic regression differed by about .007 ROC-AUC and .006 AP. The prevalence was .170, a reference for AP. These are descriptive differences from internal validation in a previously explored sample. The crash outcome is retrospective and the analysis does not evaluate prediction of future crashes.

[Original research tree](figures/research-tree.png) · [Original correlation heatmap](figures/research-heatmap.png) · [Tree refitted on the full sample](figures/decision-tree.png)

The original tree uses the 554 development cases from the earlier 70/30 split, with a depth limit of three. Its first split is on disinhibition, followed by dissociative and risky driving styles. The figure is retained to show that original fit, which used the historical imputation procedure. The current full-sample tree illustrates how splits can change when the training sample changes. Neither figure supplies the scores in the table, which come from held-out folds. The heatmap shows descriptive correlations among the five styles and five personality domains.

## Evaluation and reproducibility

Each task uses five shared folds, stratified for classification. Mean imputation and scaling, where applicable, are fitted inside the training folds. Model settings were fixed. The classification output also includes Brier score and balanced accuracy at a threshold of 0.5. Fold standard deviations describe variability across folds. The earlier research notebook documented the historical workflow and its correction for imputation before splitting. This portfolio reproduces its corrected predictive sections.

The current classification benchmark uses the 22 predictors in the original research tree across all classifiers. The faculty notebook’s shared-fold comparison used 23 predictors, including stress reduction, so its classification metrics describe a different specification.

    python -m pip install -r requirements.txt
    export TRANSITO_CSV_PATH=/path/outside/this/project/authorized.csv
    python analysis.py

The participant data stay in restricted access. The project includes code and aggregate results only. The script regenerates the comparison figure, the full-sample tree and all tables. The two original research figures are archived as supplied.

[Classification metrics](results/classification-summary.csv) · [Regression metrics](results/regression-summary.csv)

## Personality measure

Personality Inventory for DSM-5 (PID-5), originally developed by Krueger et al. (2012). Sánchez, Montes, and colleagues (2023) later reported on a shortened and modified version for use in Argentina.

Krueger, R. F., Derringer, J., Markon, K. E., Watson, D., & Skodol, A. E. (2012). Initial construction of a maladaptive personality trait model and inventory for DSM-5. *Psychological Medicine, 42*(9), 1879–1890. https://doi.org/10.1017/S0033291711002674

Sánchez, R. O., Montes, S. A., Galarza, A. L., Somerstein, L. D., & Gainza, M. D. (2023). Una versión reducida y modificada del Inventario de Personalidad para el DSM-5 (PID-5) en la Argentina. *Interdisciplinaria, 40*(1), 115–136. https://doi.org/10.16888/interd.2023.40.1.7
