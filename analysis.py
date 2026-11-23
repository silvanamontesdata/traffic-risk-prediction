import os, re, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score

SEED = 42
root = Path(__file__).resolve().parent
csv_path = Path(os.environ['TRANSITO_CSV_PATH']).resolve()
if csv_path.is_relative_to(root):
    raise ValueError('Keep the participant CSV outside the project.')
raw=pd.read_csv(csv_path,sep=';',encoding='latin1',dtype='string',keep_default_na=False)
def canon(s):
    s=unicodedata.normalize('NFKD',str(s).strip())
    s=''.join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_')
names=[canon(x) for x in raw.columns]
assert len(names)==len(set(names)), 'Colisión de nombres normalizados'
df=raw.copy(); df.columns=names
for col in df:
    df[col]=df[col].str.normalize('NFC').str.strip().replace('',pd.NA)
domains=list(map(canon,['Af.Negativa','Desapego','Antagonismo','Desinhibición','Psicoticismo']))
styles=list(map(canon,['Riesgo','Disoc','Agresion','Prudencia','Ans']))
behaviors=[canon(x) for x in raw.columns if re.match(r'^R\d+\.',x)]
facets=list(map(canon,['AfeANS','AfeDEP','AfeSUM','AfeLAB','AfeAS','AfePERS','AfeHOST','AfeDESC','DesaRetiro','DesaANH','DesaEI','DesaAR','AntGRAND','AntBA','AntMAN','AntENG','AntINS','DesiAR','DesiIMP','DesiIRR','DesiDIS','DesAPR','PsiEX','PsiDES','PsiCE']))
numeric_cols=list(dict.fromkeys(['edad']+domains+styles+behaviors+facets+['conductasriesgo']))
audit=[]
for col in numeric_cols:
    before=df[col].notna()
    converted=pd.to_numeric(df[col].str.replace(',','.',regex=False),errors='coerce')
    audit.append({'variable':col,'faltantes_entrada':int((~before).sum()),'no_convertibles':int((before & converted.isna()).sum()),'faltantes_final':int(converted.isna().sum())})
    df[col]=converted.astype(float)
# A blank response to question 7 indicates no reported event.
y_class = df['stnueva'].notna().astype(int)

specific_behaviors = [
    'r1_excedervelocidad', 'r2_alcohol', 'r3_distraido',
    'r4_celular', 'r5_cansado_maldormido',
]
crash_features = specific_behaviors + styles + domains
valid=y_class.notna(); Xc=df.loc[valid,crash_features];yc=y_class.loc[valid].astype(int)
folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(Xc,yc))
class_models={
 'Logística':Pipeline([('impute',SimpleImputer(strategy='mean')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=SEED))]),
 'Árbol':Pipeline([('impute',SimpleImputer(strategy='mean')),('model',DecisionTreeClassifier(max_depth=3,class_weight='balanced',random_state=SEED))]),
 'Random Forest':Pipeline([('impute',SimpleImputer(strategy='mean')),('model',RandomForestClassifier(n_estimators=500,max_depth=10,class_weight='balanced',random_state=SEED,n_jobs=1))]),
 'MLP':Pipeline([('impute',SimpleImputer(strategy='mean')),('scale',StandardScaler()),('model',MLPClassifier(hidden_layer_sizes=(10,5),max_iter=5000,alpha=.001,random_state=SEED))])}
from sklearn.base import clone
from sklearn.inspection import permutation_importance


def held_out_importance(model, predictors, outcome, fold, scoring):
    result = permutation_importance(
        model, predictors, outcome, scoring=scoring, n_repeats=12,
        random_state=SEED + fold,
    )
    return pd.Series(result.importances_mean, index=predictors.columns)


crash_importances = []
risk_importances = []
classification_rows=[];oof_class={};class_warnings=[]
for name,pipe in class_models.items():
    oof=np.full(len(yc),np.nan)
    for fold,(tr,te) in enumerate(folds,1):
        fit=clone(pipe)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fit.fit(Xc.iloc[tr],yc.iloc[tr])
        for w in caught:class_warnings.append({'modelo':name,'fold':fold,'categoría':w.category.__name__,'mensaje':str(w.message)})
        if name == 'Logística':
            crash_importances.append(held_out_importance(
                fit, Xc.iloc[te], yc.iloc[te], fold, 'roc_auc'
            ))
        proba=fit.predict_proba(Xc.iloc[te])[:,1];oof[te]=proba
        classification_rows.append({'modelo':name,'fold':fold,'N_train':len(tr),'N_test':len(te),'AUC':roc_auc_score(yc.iloc[te],proba),'AP':average_precision_score(yc.iloc[te],proba),'Brier':brier_score_loss(yc.iloc[te],proba),'balanced_accuracy':balanced_accuracy_score(yc.iloc[te],proba>=.5)})
    assert np.isfinite(oof).all();oof_class[name]=oof
classification_folds=pd.DataFrame(classification_rows)
classification_summary=classification_folds.groupby('modelo')[['AUC','AP','Brier','balanced_accuracy']].agg(['mean','std'])
from sklearn.dummy import DummyRegressor
valid=df.conductasriesgo.notna();yr=df.loc[valid,'conductasriesgo'];Xr=df.loc[valid]
reg_folds=list(KFold(n_splits=5,shuffle=True,random_state=SEED).split(Xr))
reg_rows=[];reg_warnings=[];reg_oof={}
reg_models={'Media':DummyRegressor(strategy='mean'),'Lineal':LinearRegression(),'Elastic Net':ElasticNet(alpha=.1,l1_ratio=.5,max_iter=10000,random_state=SEED),'Random Forest':RandomForestRegressor(n_estimators=300,min_samples_leaf=5,random_state=SEED,n_jobs=1)}
for representation,cols in [('Dominios',domains),('Facetas',facets)]:
    for name,model in reg_models.items():
        steps=[('impute',SimpleImputer(strategy='mean'))]
        if name in ['Lineal','Elastic Net']:steps.append(('scale',StandardScaler()))
        pipe=Pipeline(steps+[('model',model)]);oof=np.full(len(yr),np.nan)
        for fold,(tr,te) in enumerate(reg_folds,1):
            fit=clone(pipe)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');fit.fit(Xr.iloc[tr][cols],yr.iloc[tr])
            for w in caught:reg_warnings.append({'representación':representation,'modelo':name,'fold':fold,'mensaje':str(w.message)})
            if representation == 'Facetas' and name == 'Random Forest':
                risk_importances.append(held_out_importance(
                    fit, Xr.iloc[te][cols], yr.iloc[te], fold, 'neg_mean_absolute_error'
                ))
            pred=fit.predict(Xr.iloc[te][cols]);oof[te]=pred
            reg_rows.append({'representación':representation,'modelo':name,'fold':fold,'N_train':len(tr),'N_test':len(te),'R2':r2_score(yr.iloc[te],pred),'RMSE':np.sqrt(mean_squared_error(yr.iloc[te],pred)),'MAE':mean_absolute_error(yr.iloc[te],pred)})
        assert np.isfinite(oof).all();reg_oof[(representation,name)]=oof
regression_folds=pd.DataFrame(reg_rows)
regression_summary=regression_folds.groupby(['representación','modelo'])[['R2','RMSE','MAE']].agg(['mean','std'])

classification_folds.to_csv(root / 'results/classification-folds.csv', index=False)
regression_folds.to_csv(root / 'results/regression-folds.csv', index=False)
classification_summary.to_csv(root / 'results/classification-summary.csv')
regression_summary.to_csv(root / 'results/regression-summary.csv')

for task, importances, metric in [
    ('Self-reported crash involvement', crash_importances, 'ROC-AUC decrease'),
    ('Risky-driving score', risk_importances, 'MAE increase'),
]:
    fold_scores = pd.DataFrame(importances)
    summary = pd.DataFrame({
        'mean_change': fold_scores.mean(),
        'fold_sd': fold_scores.std(),
        'positive_folds': fold_scores.gt(0).sum(),
    }).sort_values('mean_change', ascending=False)
    summary.insert(0, 'metric', metric)
    summary.index.name = 'predictor'
    filename = 'crash-importance.csv' if task == 'Self-reported crash involvement' else 'risk-importance.csv'
    summary.to_csv(root / 'results' / filename)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), layout='constrained')
specs = [
    (axes[0], classification_summary['AUC']['mean'],
     ['Logística', 'Árbol', 'Random Forest', 'MLP'], ['Logistic', 'Tree', 'Random Forest', 'MLP'],
     'Self-reported crash involvement', 'Mean fold ROC-AUC', .5, (.45, .69)),
    (axes[1], regression_summary.loc['Facetas']['R2']['mean'],
     ['Lineal', 'Elastic Net', 'Random Forest'], ['Linear', 'Elastic Net', 'Random Forest'],
     'Risk-behavior score', 'Mean fold R²', 0, (.0, .33)),
]
for ax, series, keys, names, title, xlabel, baseline, limits in specs:
    values = series.loc[keys].values
    ax.scatter(values, range(len(keys)), color='#116b74', s=55)
    ax.set(yticks=range(len(keys)), yticklabels=names, title=title, xlabel=xlabel, xlim=limits)
    ax.invert_yaxis()
    ax.axvline(baseline, color='#8c9299', ls='--', lw=1)
    ax.grid(axis='x', alpha=.2)
    for y, val in enumerate(values):
        ax.annotate(f'{val:.3f}', (val, y), xytext=(7, 0), textcoords='offset points', va='center')
fig.suptitle('How much does model complexity add?', fontsize=16, weight='bold')
fig.savefig(root / 'figures/model-comparison.png', dpi=145)
plt.close(fig)

# Descriptive correlation matrix for the same five styles and five domains.
heatmap_labels = [
    'Risky', 'Dissociative', 'Aggressive', 'Careful', 'Anxious',
    'Negative affectivity', 'Detachment', 'Antagonism', 'Disinhibition', 'Psychoticism',
]
correlations = df[styles + domains].corr().to_numpy()
fig, ax = plt.subplots(figsize=(9.1, 7.6), layout='constrained')
heatmap = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(heatmap_labels)), heatmap_labels, rotation=40, ha='right')
ax.set_yticks(range(len(heatmap_labels)), heatmap_labels)
ax.set_title('Driving styles and personality domains', loc='left', weight='bold')
fig.colorbar(heatmap, ax=ax, shrink=.75, label='Pearson correlation')
for (row, col), value in np.ndenumerate(correlations):
    ax.text(col, row, f'{value:.2f}', ha='center', va='center',
            color='white' if abs(value) > .75 else '#192536', fontsize=9)
fig.savefig(root / 'figures/correlation-heatmap.png', dpi=170)
plt.close(fig)

# Reproduce the illustrative 70/30 split independently of the five-fold comparison.
train, test = train_test_split(np.arange(len(yc)), test_size=.30, stratify=yc, random_state=SEED)
holdout_model = clone(class_models['Árbol']).fit(Xc.iloc[train], yc.iloc[train])
holdout_auc = roc_auc_score(yc.iloc[test], holdout_model.predict_proba(Xc.iloc[test])[:, 1])
pd.DataFrame([{'N_train': len(train), 'N_test': len(test), 'AUC': holdout_auc}]).to_csv(
    root / 'results/tree-holdout.csv', index=False
)
tree = holdout_model.named_steps['model']
fig, ax = plt.subplots(figsize=(18, 7))
feature_labels = dict(zip(specific_behaviors, [
    'Speeding', 'Driving after drinking', 'Distracted driving', 'Phone use',
    'Driving while tired',
]))
feature_labels.update(zip(styles + domains, [
    'Risky driving style', 'Dissociative driving style', 'Aggressive driving style',
    'Careful driving style', 'Anxious driving style', *heatmap_labels[5:],
]))
tree_labels = [feature_labels[column] for column in Xc.columns]
plot_tree(tree, ax=ax, feature_names=tree_labels, class_names=['No', 'Yes'],
          filled=True, rounded=True, impurity=False, precision=2, fontsize=9)
ax.set_title(f'Self-reported crash involvement · trained on {len(train)} participants')
fig.savefig(root / 'figures/decision-tree.png', dpi=110, bbox_inches='tight')
plt.close(fig)
print(classification_summary.round(4))
print(regression_summary.round(4))
print('Warnings recorded:', len(class_warnings), len(reg_warnings))
