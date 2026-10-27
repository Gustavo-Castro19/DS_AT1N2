# Classificação de risco de crédito

Projeto de classificação supervisionada de `Loan_Status` usando Gaussian Naive Bayes, Árvore de Decisão e Random Forest, em versões baseline e ajustadas.

## Estrutura

- `data/realistic_credit_scoring_dataset-1.csv`: base fornecida no projeto.
- `data_fraud.ipynb`: notebook executável da análise.
- `Trabalho_Ciencia_de_Dados_Python.docx`: enunciado.
- `requirements.txt`: dependências Python.

## Como executar

Use Python 3.10 ou superior. Na raiz do projeto:

```bash
python -m pip install -r requirements.txt
jupyter notebook data_fraud.ipynb
```

Execute todas as células em ordem. O notebook usa divisão estratificada, semente 42, validação cruzada de cinco folds e mantém a imputação dentro do pipeline. `Default_Probability` foi excluída por sua associação muito forte com o alvo e pela falta de documentação sobre sua origem.

## Dados e limitações

Fonte: [Chaitanya Jamble, Realistic Credit Scoring & Loan Default Prediction (Kaggle)](https://www.kaggle.com/datasets/chaitanyajamble/realistic-credit-scoring-and-loan-default-prediction). O diretório `data/` inclui uma licença CC0. A página e os arquivos locais não esclarecem o método de geração ou a representatividade dos registros; esses pontos devem ser apresentados como limitações no relatório. A variável `Default_Probability` foi removida da análise principal por essa incerteza; só deve voltar numa análise de sensibilidade se a fonte confirmar que é independente do alvo e disponível no momento da decisão.
