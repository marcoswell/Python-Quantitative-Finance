# Dashboard CARR (Conditional Autoregressive Range)

Gerado a partir do notebook `modelo_carr_v1.ipynb`.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## O que o dashboard faz

- **Dados & Range**: baixa preços (Yahoo Finance) ou lê um CSV enviado, calcula amplitude
  log e volatilidade anualizada.
- **Estimação CARR**: estima ω, α, β por máxima verossimilhança (mesma especificação do
  notebook) e mostra o ajuste CARR(1,1) vs. range observado, além dos resíduos εₜ.
- **Modelo Adaptativo**: reproduz a recalibragem mensal com janela móvel (padrão 30 dias),
  configurável na barra lateral.
- **Regimes de Mercado**: classifica dias em "Expansão" / "Estresse-Contração" com base no
  choque padronizado zₜ = Rₜ/ψₜ e no sinal do retorno diário; limiares ajustáveis.
- **VaR**: VaR paramétrico dinâmico (aproximação de Parkinson) com backtest de violações
  (hit rate) e histograma.
- **Monte Carlo**: simulação de PnL para VaR 90/95/99% dado um capital investido.

## Observações

- Todos os cálculos pesados (otimização MLE, modelo adaptativo) são cacheados com
  `st.cache_data`, então trocar apenas parâmetros visuais não reprocessa tudo.
- Se preferir não depender do Yahoo Finance, use a opção "Upload CSV" na barra lateral —
  o arquivo precisa ter colunas `Open, High, Low, Close` e uma coluna de data como índice.
