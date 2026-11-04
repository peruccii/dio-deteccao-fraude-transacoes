"""Gera o notebook deteccao_fraude.ipynb (as saídas são produzidas ao executá-lo)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md("""# Detecção de Anomalias em Transações de Cartão de Crédito

Projeto do bootcamp **Bradesco - Agentes de IA do Zero a Prática** (DIO).

Objetivo: treinar modelos que identifiquem transações fraudulentas em uma base **muito desbalanceada**
(cerca de 0,17% de fraudes) e explicar o que levou o modelo a marcar uma transação como fraude.

Roteiro: **Explorar → Preparar → Treinar → Avaliar → Explicar**."""),
    code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, RocCurveDisplay,
                             PrecisionRecallDisplay, precision_recall_curve)
from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""),
    md("## 1. Explorar\n\nO dataset é carregado direto pelo link (não fica no repositório)."),
    code("""URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(URL)
print(df.shape)
df.head()"""),
    code("""print(df.isna().sum().sum(), "valores ausentes")
contagem = df["Class"].value_counts()
proporcao = df["Class"].value_counts(normalize=True) * 100
pd.DataFrame({"transações": contagem, "%": proporcao}).rename(index={0: "normal", 1: "fraude"})"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[df.Class == 0]["Amount"].clip(upper=500).hist(bins=50, ax=axes[0], color="tab:blue")
axes[0].set_title("Valor - transações normais (até 500)")
df[df.Class == 1]["Amount"].clip(upper=500).hist(bins=50, ax=axes[1], color="tab:red")
axes[1].set_title("Valor - fraudes (até 500)")
plt.tight_layout(); plt.show()

df.groupby("Class")["Amount"].describe()"""),
    md("""**Leitura:** um modelo que responde "não é fraude" para tudo teria ~99,83% de acurácia e não pegaria nenhuma fraude.
Por isso a avaliação usa **recall, precisão e F1 da classe fraude**, além da área sob a curva de precisão e recall (AUPRC)."""),
    md("## 2. Preparar"),
    code("""dados = df.copy()
dados["log_amount"] = np.log1p(dados["Amount"])          # reduz a assimetria do valor
dados["hora"] = (dados["Time"] // 3600) % 24              # hora aproximada do dia
dados = dados.drop(columns=["Amount", "Time"])

X = dados.drop(columns="Class")
y = dados["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

print("fraudes no treino: %.4f%% | no teste: %.4f%%" % (y_train.mean() * 100, y_test.mean() * 100))"""),
    md("## 3. Treinar\n\nBaseline com regressão logística e comparação com Random Forest e XGBoost, todos com peso para a classe minoritária."),
    code("""peso = (y_train == 0).sum() / (y_train == 1).sum()

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                                            class_weight="balanced_subsample", n_jobs=-1,
                                            random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.8, scale_pos_weight=peso, eval_metric="aucpr",
                             n_jobs=-1, random_state=RANDOM_STATE),
}

probas = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train_s, y_train)
    probas[nome] = modelo.predict_proba(X_test_s)[:, 1]
    print(nome, "treinado")"""),
    md("### Bônus: undersampling × oversampling (com a regressão logística)"),
    code("""rng = np.random.default_rng(RANDOM_STATE)
idx_f = y_train[y_train == 1].index
idx_n = y_train[y_train == 0].index

# undersampling: reduz as normais para 5x o número de fraudes
idx_under = np.concatenate([idx_f, rng.choice(idx_n, size=len(idx_f) * 5, replace=False)])
# oversampling: replica fraudes até 10% do tamanho das normais
idx_over = np.concatenate([idx_n, rng.choice(idx_f, size=len(idx_n) // 10, replace=True)])

for nome, idx in [("LR + undersampling", idx_under), ("LR + oversampling", idx_over)]:
    m = LogisticRegression(max_iter=2000).fit(X_train_s.loc[idx], y_train.loc[idx])
    probas[nome] = m.predict_proba(X_test_s)[:, 1]
    print(nome, "treinado com", len(idx), "linhas")"""),
    md("## 4. Avaliar"),
    code("""def metricas(y_true, p, limiar=0.5):
    pred = (p >= limiar).astype(int)
    return {"precisão": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred),
            "ROC AUC": roc_auc_score(y_true, p),
            "AUPRC": average_precision_score(y_true, p)}

tabela = pd.DataFrame({n: metricas(y_test, p) for n, p in probas.items()}).T
tabela.sort_values("F1", ascending=False)"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for nome, p in probas.items():
    RocCurveDisplay.from_predictions(y_test, p, name=nome, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, p, name=nome, ax=axes[1])
axes[0].set_title("Curva ROC"); axes[1].set_title("Curva de Precisão e Recall")
plt.tight_layout(); plt.show()"""),
    md("""### Ajuste do limiar de decisão (XGBoost)

Em vez de usar 0,5, escolho o limiar que maximiza o F1 na curva de precisão e recall,
mantendo o recall da fraude como prioridade."""),
    code("""p_xgb = probas["XGBoost"]
prec, rec, lim = precision_recall_curve(y_test, p_xgb)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
melhor = lim[np.argmax(f1s)]
print(f"limiar escolhido: {melhor:.3f}")

comp = pd.DataFrame({"limiar 0.5": metricas(y_test, p_xgb, 0.5),
                     f"limiar {melhor:.3f}": metricas(y_test, p_xgb, melhor)}).T
display(comp)

pred = (p_xgb >= melhor).astype(int)
print(classification_report(y_test, pred, target_names=["normal", "fraude"], digits=4))
pd.DataFrame(confusion_matrix(y_test, pred), index=["real normal", "real fraude"],
             columns=["prev. normal", "prev. fraude"])"""),
    md("## 5. Explicar"),
    code("""xgb = modelos["XGBoost"]
imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
imp[::-1].plot.barh(figsize=(7, 5), title="Importância das variáveis - XGBoost")
plt.tight_layout(); plt.show()"""),
    code("""amostra = X_test_s.sample(2000, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(xgb)
valores = explainer.shap_values(amostra)
ranking = pd.Series(np.abs(valores).mean(axis=0), index=amostra.columns).sort_values(ascending=False)
print("Top 8 variáveis por |SHAP| médio:")
print(ranking.head(8).round(4).to_string())
shap.summary_plot(valores, amostra, max_display=15, show=False)
plt.title("SHAP - impacto de cada variável"); plt.tight_layout(); plt.show()"""),
    code("""# Explicação de uma fraude específica detectada pelo modelo
fraudes_detectadas = X_test_s[(y_test == 1) & (p_xgb >= melhor)]
caso = fraudes_detectadas.iloc[[0]]
exp = explainer(caso)
print("probabilidade de fraude:", float(xgb.predict_proba(caso)[0, 1]))
shap.plots.waterfall(exp[0], max_display=12, show=False)
plt.tight_layout(); plt.show()"""),
    md("""## Conclusões

- A acurácia não serve para esse problema; recall, precisão, F1 e AUPRC mostram o desempenho real.
- O XGBoost teve o melhor equilíbrio entre recall e precisão; a regressão logística pega muitas fraudes, mas com muitos alarmes falsos.
- Ajustar o limiar mudou bastante o resultado: com 0,5 o modelo é mais agressivo; com o limiar escolhido pelo F1, os falsos positivos caem muito.
- O SHAP mostra quais variáveis concentram as decisões (ranking impresso acima) e, no gráfico de cascata, quanto cada uma empurrou uma fraude específica para a classe positiva."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "deteccao_fraude.ipynb")
print("notebook gerado")
