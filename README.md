# Classificação de risco de crédito

Projeto de classificação supervisionada de `Loan_Status` usando Gaussian Naive Bayes, Árvore de Decisão e Random Forest, em versões baseline e ajustadas.

## Estrutura

- `data/realistic_credit_scoring_dataset-1.csv`: base fornecida no projeto.
- `notebooks/data_fraud.ipynb`: notebook executável da análise.
- `relatorio.pdf` e `relatorio.html`: relatório e fonte editável.
- `resultados/`: gráficos PNG gerados pelo notebook.
- `requirements.txt`: dependências Python.

## Como executar

Use Python 3.10 ou superior. Na raiz do projeto:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/data_fraud.ipynb
```

Execute todas as células em ordem. O notebook cria `resultados/` na raiz do projeto e salva nela os gráficos PNG. O notebook usa divisão estratificada, semente 42, validação cruzada de cinco folds e mantém a imputação dentro do pipeline. `Default_Probability` foi excluída da análise principal por sua associação muito forte com o alvo e incerteza de disponibilidade no momento da previsão.

## Dados e limitações

Fonte: [Chaitanya Jamble, Realistic Credit Scoring & Loan Default Prediction (Kaggle)](https://www.kaggle.com/datasets/chaitanyajamble/realistic-credit-scoring-and-loan-default-prediction). O diretório `data/` inclui uma licença CC0. A descrição do dataset informa que são 15 mil registros sintéticos, financeiramente realistas, com relações estruturadas por lógica de pontuação ponderada inspirada em sistemas FICO. É uma base estática e representa um cenário simulado, não uma amostra de tomadores reais. Os resultados não devem ser generalizados para crédito real sem validação externa.
