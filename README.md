# Detecção de Anomalias em Transações (Fraude em Cartão de Crédito)

Projeto do bootcamp **Bradesco - Agentes de IA do Zero a Prática**, da [DIO](https://www.dio.me/).

Notebook: [`deteccao_fraude.ipynb`](./deteccao_fraude.ipynb). Todas as saídas (tabelas e curvas) ficaram salvas.

## O problema

A base tem **284.807 transações** de cartão. Só **492 são fraude (0,17%)**, e o restante (99,83%) é normal. As colunas são `Time`, `Amount`, `Class` e `V1` a `V28`, que vêm de uma transformação PCA para proteger a privacidade.

Com um desbalanceamento desse tamanho, **a acurácia engana**. Um modelo que responde "não é fraude" para tudo acerta 99,83% e não detecta nada. Por isso a avaliação se concentra na classe fraude, com **recall** (quantas fraudes o modelo pega), **precisão** (quantos alertas são fraude de verdade), **F1** e **AUPRC** (área sob a curva de precisão e recall).

O dataset é carregado pelo link público, direto no notebook, e não fica no repositório.

## Preparação dos dados

- `log_amount = log(1 + Amount)`, para reduzir a assimetria do valor.
- `hora = (Time // 3600) % 24`, a hora aproximada do dia. As colunas originais `Time` e `Amount` foram removidas.
- Divisão treino/teste (75/25) com `stratify=y`, mantendo 0,17% de fraudes nos dois conjuntos.
- `StandardScaler` ajustado só no treino, para evitar vazamento de dados.

## Comparação dos modelos (classe fraude, limiar 0,5)

| Modelo | Precisão | Recall | F1 | ROC AUC | AUPRC |
|---|---|---|---|---|---|
| **XGBoost** (`scale_pos_weight`) | **0,853** | 0,805 | **0,829** | **0,984** | **0,853** |
| Random Forest (`class_weight`) | 0,850 | 0,780 | 0,814 | 0,980 | 0,815 |
| Regressão Logística + oversampling | 0,443 | 0,854 | 0,583 | 0,971 | 0,714 |
| Regressão Logística + undersampling | 0,216 | 0,854 | 0,344 | 0,969 | 0,682 |
| Regressão Logística (`class_weight`, baseline) | 0,059 | **0,886** | 0,110 | 0,972 | 0,706 |

A regressão logística com peso de classe tem o maior recall, mas quase todos os alertas dela são falsos (precisão de 6%). O XGBoost tem o melhor equilíbrio entre os dois.

## Limiar de decisão

No XGBoost, escolhi o limiar que **maximiza o F1** na curva de precisão e recall: **0,953**.

| Limiar | Precisão | Recall | F1 |
|---|---|---|---|
| 0,5 | 0,853 | 0,805 | 0,829 |
| **0,953** | **0,951** | 0,780 | **0,857** |

No teste, com o limiar escolhido, o modelo pegou **96 das 123 fraudes** e gerou só **5 falsos alarmes** em mais de 71 mil transações normais. Se a prioridade for recall máximo, um limiar menor pega mais fraudes em troca de mais alertas. O notebook traz as duas curvas para essa decisão.

## O que o SHAP mostrou

Ranking por |SHAP| médio, numa amostra de 2.000 transações do teste:

| Variável | V14 | V4 | V12 | V10 | V11 | V3 | V8 | V19 |
|---|---|---|---|---|---|---|---|---|
| \|SHAP\| médio | 1,88 | 1,73 | 1,03 | 0,84 | 0,61 | 0,54 | 0,42 | 0,39 |

- **V14, V4, V12 e V10** concentram a maior parte das decisões.
- No *summary plot*, valores **baixos de V14, V12 e V10** e valores **altos de V4 e V11** empurram a previsão para fraude.
- O gráfico de cascata (*waterfall*) explica uma fraude específica detectada (probabilidade de 0,999) e mostra quanto cada variável contribuiu.

## O que mudei em relação ao caminho da Expert

- Criei a variável `hora`, além do log do valor.
- Comparei **undersampling e oversampling** com o peso de classe na mesma regressão logística.
- Escolhi o limiar pelo máximo F1 na curva de precisão e recall, em vez de manter 0,5.
- Usei a **AUPRC** como métrica de comparação, porque é mais informativa que a ROC AUC em bases muito desbalanceadas.

## Como rodar

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib jupyter
jupyter notebook deteccao_fraude.ipynb
```

O arquivo `build_notebook.py` gera o notebook a partir do código, o que facilita versionar as células.
