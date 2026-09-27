"""
Dashboard CARR (Conditional Autoregressive Range) Model
=========================================================
Gerado a partir do notebook modelo_carr_v1.ipynb.

Rodar com:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CARR Model Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Funções do modelo (extraídas do notebook)
# ----------------------------------------------------------------------------

def carr_recursion(params, R):
    """Recursão do CARR(1,1): psi_t = omega + alpha*R_{t-1} + beta*psi_{t-1}"""
    omega, alpha, beta = params
    R = np.asarray(R)
    n = len(R)
    psi = np.zeros(n)
    psi[0] = np.mean(R)
    for t in range(1, n):
        psi[t] = omega + alpha * R[t - 1] + beta * psi[t - 1]
    return psi


def negative_log_likelihood(params, R):
    omega, alpha, beta = params
    psi = carr_recursion(params, R)
    psi = np.maximum(psi, 1e-8)
    log_likelihood = np.sum(-np.log(psi) - np.asarray(R) / psi)
    return -log_likelihood


def estimate_carr(R, initial_params):
    bnds = (
        (1e-6, None),   # omega > 0
        (1e-6, 0.999),  # alpha
        (1e-6, 0.999),  # beta
    )
    cons = ({
        "type": "ineq",
        "fun": lambda x: 0.999 - (x[1] + x[2]),
    })
    result = minimize(
        negative_log_likelihood,
        x0=initial_params,
        args=(R,),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
    )
    return result


def run_adaptive_carr(df, R, initial_params, window):
    """Modelo adaptativo: recalibra no início de cada mês, usando janela móvel."""
    forecasts = []
    psi_rolling = []
    params_history = []      # (omega, alpha, beta) vigentes em cada dia t
    recalibrated_flags = []  # True nos dias em que uma nova otimização ocorreu
    current_params = initial_params
    bounds = ((1e-6, None), (1e-6, 0.999), (1e-6, 0.999))

    for t in range(window, len(df)):
        R_train = R.iloc[t - window:t] if isinstance(R, pd.Series) else R[t - window:t]
        did_recalibrate = False

        if t == window or df.index[t].month != df.index[t - 1].month:
            result = minimize(
                negative_log_likelihood,
                current_params,
                args=(R_train,),
                method="L-BFGS-B",
                bounds=bounds,
            )
            omega_cand, alpha_cand, beta_cand = result.x
            if result.success and (alpha_cand + beta_cand < 0.999):
                current_params = result.x
                did_recalibrate = True

        omega, alpha, beta = current_params
        last_range = R.iloc[t - 1] if isinstance(R, pd.Series) else R[t - 1]

        psi_train = carr_recursion(current_params, R_train)
        last_psi = psi_train[-1]
        psi_rolling.append(last_psi)

        forecast = omega + alpha * last_range + beta * last_psi
        forecasts.append(forecast)
        params_history.append((omega, alpha, beta))
        recalibrated_flags.append(did_recalibrate)

    return forecasts, psi_rolling, params_history, recalibrated_flags


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Baixando dados...")
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        return df
    df.columns = df.columns.get_level_values(0)

    df["return"] = df["Close"].pct_change(1)
    df["Returns"] = np.log(df["Close"].pct_change(1))
    df["vol"] = df["return"].rolling(21).std() * np.sqrt(254)

    df["range"] = (np.log(df["High"]) - np.log(df["Low"])) * 100
    df["range_n"] = df["High"] - df["Low"]
    df = df[df["range"] > 0].copy()

    return df


@st.cache_data(show_spinner=False)
def load_data_from_csv(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no CSV: {missing}")

    df["return"] = df["Close"].pct_change(1)
    df["Returns"] = np.log(df["Close"].pct_change(1))
    df["vol"] = df["return"].rolling(21).std() * np.sqrt(254)

    df["range"] = (np.log(df["High"]) - np.log(df["Low"])) * 100
    df["range_n"] = df["High"] - df["Low"]
    df = df[df["range"] > 0].copy()

    return df


@st.cache_data(show_spinner="Estimando parâmetros CARR(1,1)...")
def cached_estimate(R_values, initial_params):
    R = pd.Series(R_values)
    result = estimate_carr(R, np.array(initial_params))
    return result.x, result.success


@st.cache_data(show_spinner="Rodando modelo adaptativo (recalibragem mensal)...")
def cached_adaptive(df_index, R_values, initial_params, window):
    df_idx = pd.DataFrame(index=pd.DatetimeIndex(df_index))
    R = pd.Series(R_values, index=df_idx.index)
    forecasts, psi_rolling, params_history, recalibrated_flags = run_adaptive_carr(
        df_idx, R, np.array(initial_params), window
    )
    return forecasts, psi_rolling, params_history, recalibrated_flags


# ----------------------------------------------------------------------------
# Sidebar — parâmetros
# ----------------------------------------------------------------------------

st.sidebar.header("⚙️ Parâmetros")

data_source = st.sidebar.radio("Fonte de dados", ["Yahoo Finance", "Upload CSV"])

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Início", value=pd.to_datetime("2020-01-01"))
    end_date = col2.date_input("Fim", value=pd.to_datetime("2025-01-01"))
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader(
        "CSV com colunas: Date (índice), Open, High, Low, Close", type=["csv"]
    )
    ticker = st.sidebar.text_input("Nome/Ticker (rótulo para os gráficos)", value="Ativo")

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros iniciais (chute)")
omega0 = st.sidebar.number_input("omega₀", value=0.7, step=0.05, format="%.4f")
alpha0 = st.sidebar.number_input("alpha₀", value=0.22, step=0.05, format="%.4f")
beta0 = st.sidebar.number_input("beta₀", value=0.7, step=0.05, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Modelo adaptativo")
window = st.sidebar.slider("Janela móvel (dias)", min_value=10, max_value=90, value=30, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Detecção de regimes")
z_long = st.sidebar.number_input("Limiar z — expansão (z >)", value=1.5, step=0.1)
z_short = st.sidebar.number_input("Limiar z — contração (z <)", value=0.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulação Monte Carlo (VaR)")
capital = st.sidebar.number_input("Capital investido (R$)", value=1_000_000, step=50_000)
n_sims = st.sidebar.number_input("Nº de simulações", value=100_000, step=10_000)

run_button = st.sidebar.button("🚀 Rodar modelo", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Corpo principal
# ----------------------------------------------------------------------------

st.title("📈 Modelo CARR (Conditional Autoregressive Range)")
st.caption(
    "ψₜ = ω + α·Rₜ₋₁ + β·ψₜ₋₁ — range condicional esperado, "
    "com resíduos εₜ ~ Exponencial(1)"
)

if not run_button and "df" not in st.session_state:
    st.info("Configure os parâmetros na barra lateral e clique em **🚀 Rodar modelo**.")
    st.stop()

# --- Carregamento de dados ---
if run_button:
    if data_source == "Yahoo Finance":
        df = load_data(ticker, str(start_date), str(end_date))
    else:
        if uploaded_file is None:
            st.warning("Envie um arquivo CSV para continuar.")
            st.stop()
        df = load_data_from_csv(uploaded_file)

    if df.empty:
        st.error("Nenhum dado retornado. Verifique o ticker/datas ou o arquivo enviado.")
        st.stop()

    st.session_state["df"] = df
    st.session_state["ticker"] = ticker
else:
    df = st.session_state["df"]
    ticker = st.session_state["ticker"]

# ----------------------------------------------------------------------------
# Estimação do CARR "cheio" (amostra inteira)
# ----------------------------------------------------------------------------

R = df["range"].copy()
initial_params = np.array([omega0, alpha0, beta0])

params_est, success = cached_estimate(tuple(R.values), tuple(initial_params))
omega_est, alpha_est, beta_est = params_est

psi = carr_recursion(params_est, R)
df["carr"] = psi

range_vals = df["range"].to_numpy()
n = len(df)
carr_est = np.zeros(n)
carr_est[0] = range_vals[0]
for t in range(1, n):
    carr_est[t] = omega_est + alpha_est * range_vals[t - 1] + beta_est * carr_est[t - 1]
df["carr_est"] = carr_est

last_range = R.iloc[-1]
last_psi = psi[-1]
forecast_range = omega_est + alpha_est * last_range + beta_est * last_psi

df["epsilon"] = df["range"] / df["carr"]

TEORIA_MD = r"""
## 1. Motivação: por que modelar a *Amplitude* (Range) em vez do retorno?

O desvio-padrão de retornos é a medida de volatilidade mais usada, mas ela descarta
informação: dentro de um único dia de negociação, o preço pode ter oscilado muito
entre a máxima e a mínima, mesmo que o fechamento tenha voltado perto da abertura.
A **amplitude (range) intradiária** — a distância entre o maior e o menor preço do
dia — captura esse movimento "escondido" e é reconhecida na literatura (Parkinson,
1980; Alizadeh, Brandt & Diebold, 2002) como um **estimador de volatilidade mais
eficiente** do que o retorno ao quadrado, para uma mesma quantidade de dados.

## 2. Definindo o Range

Define-se o range logarítmico do dia $t$ como:

$$ R_t = \ln(H_t) - \ln(L_t) $$

onde $H_t$ e $L_t$ são, respectivamente, a máxima e a mínima do dia. No código,
multiplicamos por 100 apenas para trabalhar em uma escala numérica mais confortável
para o otimizador:

$$ R_t^{(\%)} = 100 \times \big[\ln(H_t) - \ln(L_t)\big] $$

Como $H_t \geq L_t$ sempre, $R_t \geq 0$ por construção — o range é uma série
estritamente não-negativa, o que descarta modelos gaussianos "puros" (que permitem
valores negativos) e pede uma modelagem específica para variáveis positivas.

## 3. O Modelo CARR(1,1)

O **CARR — Conditional Autoregressive Range** (Chou, 2005) é a adaptação do
GARCH de Engle e Bollerslev para uma variável estritamente positiva. A ideia é a
mesma do GARCH: o valor esperado hoje depende do valor observado ontem e da própria
expectativa de ontem — só que aplicada ao range em vez de à variância.

A especificação CARR(1,1) é:

$$ \psi_t = \omega + \alpha R_{t-1} + \beta \psi_{t-1} $$

onde:

- $\psi_t = E[R_t \mid \mathcal{F}_{t-1}]$ é o **range condicional esperado** (a
  "previsão" do modelo para a amplitude de hoje, dado tudo que se sabe até ontem);
- $\omega > 0$ é um termo constante (nível de longo prazo);
- $\alpha \geq 0$ captura o quanto o choque de ontem ($R_{t-1}$) se propaga para hoje;
- $\beta \geq 0$ captura a persistência — o quanto a própria expectativa de ontem
  carrega inércia para hoje.

Para o processo ser bem definido e estacionário (não explodir ao longo do tempo),
exige-se:

$$ \omega > 0, \quad 0 \le \alpha < 1, \quad 0 \le \beta < 1, \quad \alpha + \beta < 1 $$

Essas são exatamente as *bounds* e a *constraint* usadas no otimizador do código.

## 4. Distribuição condicional e log-verossimilhança

Para estimar $\omega, \alpha, \beta$ precisamos de uma distribuição para o range.
O CARR assume que o range é o produto entre a expectativa condicional $\psi_t$ e um
choque multiplicativo positivo $\epsilon_t$:

$$ R_t = \psi_t \, \epsilon_t, \qquad \epsilon_t \sim \text{Exponencial}(1) $$

Como $\epsilon_t$ tem média 1, isso garante $E[R_t \mid \mathcal{F}_{t-1}] = \psi_t$,
consistente com a definição de $\psi_t$. A densidade condicional do range é então:

$$ f(R_t \mid \mathcal{F}_{t-1}) = \frac{1}{\psi_t} \exp\!\left(-\frac{R_t}{\psi_t}\right) $$

A log-verossimilhança da amostra completa é a soma dos logs dessa densidade:

$$ \log L = \sum_t \left[-\log(\psi_t) - \frac{R_t}{\psi_t}\right] $$

## 5. Estimação por Máxima Verossimilhança (MLE)

Os parâmetros $(\omega, \alpha, \beta)$ são escolhidos como os que **maximizam**
$\log L$. Como o `scipy.optimize.minimize` só minimiza, o código minimiza o negativo
da log-verossimilhança:

$$ (\hat\omega, \hat\alpha, \hat\beta) = \arg\min_{\omega,\alpha,\beta} \; -\log L(\omega,\alpha,\beta \mid R_1,\dots,R_T) $$

O algoritmo recalcula $\psi_t$ recursivamente (via `carr_recursion`) a cada tentativa
de parâmetros, e o otimizador (SLSQP, com bounds e a restrição de estacionariedade
$\alpha+\beta<1$) busca o conjunto de parâmetros que melhor explica a série
observada de ranges.

## 6. Previsão (Forecast)

Uma vez estimados $\hat\omega, \hat\alpha, \hat\beta$, a previsão de amplitude para o
próximo período é direta — basta aplicar a própria equação do modelo usando o
último range observado e o último $\psi$ calculado:

$$ \hat\psi_{t+1} = \hat\omega + \hat\alpha R_t + \hat\beta \hat\psi_t $$

## 7. Resíduos padronizados

O resíduo do modelo é definido como:

$$ \hat\epsilon_t = \frac{R_t}{\hat\psi_t} $$

Como $E[\epsilon_t] = 1$ por construção, um modelo bem ajustado deve produzir
resíduos com média próxima de 1. A interpretação é direta:

- $\hat\epsilon_t > 1$: o range observado foi **maior** que o esperado (choque de
  volatilidade positivo, mercado mais "agitado" que o previsto);
- $\hat\epsilon_t < 1$: o range observado foi **menor** que o esperado (mercado mais
  "calmo" que o previsto).

## 8. Modelo Adaptativo (janela móvel com recalibragem mensal)

Mercados mudam de regime ao longo do tempo, e os parâmetros $(\omega,\alpha,\beta)$
estimados com uma amostra fixa podem ficar defasados. A versão **adaptativa**
resolve isso:

1. Usa apenas uma **janela móvel** das últimas $W$ observações (padrão: 30 dias) para
   reestimar o modelo;
2. **Recalibra os parâmetros no início de cada mês** (quando o mês da data muda em
   relação ao dia anterior), em vez de a cada novo dado — um compromisso entre
   estabilidade numérica e capacidade de adaptação;
3. Inclui uma **trava de segurança**: só aceita os novos parâmetros se a otimização
   convergiu (`result.success`) e se a condição de estacionariedade
   $\alpha+\beta<0.999$ continuar satisfeita; caso contrário, mantém os parâmetros do
   mês anterior, evitando que o modelo "exploda" numericamente.

O resultado é uma série $\psi_t^{\text{adap}}$ que reage mais rápido a mudanças
estruturais de volatilidade do que a versão estimada uma única vez com a amostra
inteira.

## 9. Detecção de Regimes de Mercado

Combinando o choque padronizado do CARR adaptativo,

$$ z_t = \frac{R_t}{\psi_t^{\text{adap}}}, $$

com o sinal do **retorno diário** $r_t$, é possível classificar cada dia em um
regime bidimensional (direção × intensidade do movimento):

| Retorno | Range/z | Regime | Leitura |
|---|---|---|---|
| Positivo ↑ | Baixo ↓ | Tendência Calma | Alta sustentável, baixo ruído |
| Positivo ↑ | Alto ↑ | Expansão | Movimento comprador forte |
| Negativo ↓ | Alto ↑ | Estresse | Pânico / *sell-off* |
| Negativo ↓ | Baixo ↓ | Contração | Realização de lucros / exaustão |

No dashboard, dias com $r_t \geq 0$ e $z_t$ acima de um limiar são marcados como
**Expansão**; dias com $r_t < 0$ e $z_t$ abaixo de um limiar são marcados como
**Estresse/Contração**. Essa classificação pode alimentar regras de alocação de
risco (ex.: reduzir exposição em regimes de Estresse).

## 10. Do Range à Volatilidade: o estimador de Parkinson

Para usar o CARR em gestão de risco (VaR), é preciso converter a **previsão de
range** em uma **previsão de volatilidade dos retornos**. O estimador de Parkinson
(1980) relaciona os dois:

$$ \sigma_t^2 \approx \frac{1}{4\ln(2)} \left[\ln\!\left(\frac{H_t}{L_t}\right)\right]^2 $$

Como o CARR modela exatamente $E[R_t\mid\mathcal F_{t-1}] = \psi_t$, com
$R_t = \ln(H_t/L_t)$, uma aproximação natural para a volatilidade condicional é:

$$ \hat\sigma_t^2 \approx \frac{\psi_t^2}{4\ln(2)} \qquad\Longrightarrow\qquad \hat\sigma_t \approx \frac{\psi_t}{\sqrt{4\ln(2)}} $$

(no código, dividimos $\psi_t$ por 100 antes, pois o range foi originalmente
multiplicado por 100 na etapa de construção dos dados).

## 11. VaR paramétrico dinâmico

Com $\hat\sigma_t$ em mãos, o **Value at Risk** paramétrico (assumindo retornos
condicionalmente normais, média zero) para o nível de confiança $c$ é:

$$ \text{VaR}_t^{(c)} = z_{c} \cdot \hat\sigma_t $$

onde $z_c$ é o quantil da normal padrão (ex.: $z_{0.95} \approx -1{,}645$,
$z_{0.99} \approx -2{,}326$). O resultado é um VaR **dinâmico**: ele varia dia a
dia conforme a previsão de amplitude do CARR muda, em vez de usar um desvio-padrão
fixo estimado uma única vez.

### Backtesting (taxa de violação / *hit rate*)

Para validar o modelo, compara-se o retorno realizado com o VaR previsto: uma
**violação** ocorre quando o retorno realizado é pior (mais negativo) que o VaR:

$$ \text{Violação}_t = \mathbb{1}\{ r_t < \text{VaR}_t^{(c)} \} $$

Um modelo bem calibrado deve produzir uma taxa de violações próxima do nível
nominal — por exemplo, cerca de 5% de dias violados para o VaR de 95%, e cerca de
1% para o VaR de 99%.

## 12. VaR via Simulação de Monte Carlo

Como alternativa (ou complemento) ao VaR paramétrico fechado, pode-se **simular**
milhares de cenários de retorno usando a volatilidade projetada pelo CARR como
insumo:

1. Toma-se a última volatilidade condicional projetada, $\hat\sigma_T$;
2. Simulam-se $N$ retornos aleatórios: $\tilde r_i \sim \mathcal N(0, \hat\sigma_T^2)$,
   $i = 1,\dots,N$;
3. Convertem-se os retornos simulados em resultado financeiro (PnL), dado um
   capital investido $C$: $\widetilde{PnL}_i = C \cdot \tilde r_i$;
4. O VaR ao nível $c$ é o percentil correspondente da distribuição simulada de PnL
   (ex.: percentil 5 para VaR 95%, percentil 1 para VaR 99%).

A vantagem da simulação é a flexibilidade: é possível trocar a distribuição do
choque (não precisa ser normal), incorporar caudas mais pesadas, correlações entre
ativos, ou horizontes de mais de um dia — mantendo o CARR como o motor que fornece a
estimativa dinâmica de volatilidade em cada instante.

## 13. Resumo do fluxo

$$
\underbrace{H_t, L_t}_{\text{dados de preço}}
\;\rightarrow\;
\underbrace{R_t = \ln(H_t/L_t)}_{\text{range}}
\;\rightarrow\;
\underbrace{\psi_t = \omega+\alpha R_{t-1}+\beta\psi_{t-1}}_{\text{CARR(1,1), via MLE}}
\;\rightarrow\;
\underbrace{\hat\sigma_t \approx \psi_t/\sqrt{4\ln 2}}_{\text{volatilidade (Parkinson)}}
\;\rightarrow\;
\underbrace{\text{VaR}_t = z_c\,\hat\sigma_t}_{\text{risco paramétrico ou Monte Carlo}}
$$

Esse é o encadeamento teórico completo, do range bruto até a métrica de risco final,
que sustenta todas as abas deste dashboard.
"""

tabs = st.tabs([
    "📖 Teoria",
    "📊 Dados & Range",
    "🧮 Estimação CARR(1,1)",
    "🔄 Modelo Adaptativo",
    "🚦 Regimes de Mercado",
    "⚠️ VaR",
    "🎲 Monte Carlo",
])

# --- Aba 0: Teoria ---
with tabs[0]:
    st.markdown(TEORIA_MD)

# --- Aba 1: Dados & Range ---
with tabs[1]:
    st.subheader(f"Preço e Amplitude — {ticker}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Observações", f"{len(df):,}".replace(",", "."))
    c2.metric("Range médio (log·100)", f"{df['range'].mean():.3f}")
    c3.metric("Vol. anualizada (última)", f"{df['vol'].dropna().iloc[-1]:.2%}" if df["vol"].notna().any() else "n/d")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Amplitude Logarítmica", "Amplitude Normal", "Volatilidade Anualizada"),
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["range"], name="Amplitude Log", line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["range_n"], name="Amplitude Normal", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["vol"], name="Vol. Anualizada", line=dict(color="blue")), row=3, col=1)
    fig.update_layout(height=700, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver dados brutos"):
        st.dataframe(df.tail(200), use_container_width=True)

# --- Aba 2: Estimação CARR ---
with tabs[2]:
    st.subheader("Parâmetros estimados (amostra completa)")

    if not success:
        st.warning("O otimizador não convergiu com sucesso — resultados podem ser instáveis.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ω (omega)", f"{omega_est:.6f}")
    c2.metric("α (alpha)", f"{alpha_est:.6f}")
    c3.metric("β (beta)", f"{beta_est:.6f}")
    c4.metric("α + β", f"{alpha_est + beta_est:.6f}",
              help="Deve ser < 1 para estacionariedade")

    st.metric("Previsão de range para o próximo período", f"{forecast_range:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["range"], name="Range", line=dict(color="gray", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["carr"], name="CARR(1,1)", line=dict(color="red", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["carr_est"].shift(1), name="Estimativa", line=dict(color="blue", width=1.5)))
    fig.update_layout(
        title="Comparação: Range vs CARR(1,1) vs Estimativa",
        xaxis_title="Data", yaxis_title="Range",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resíduos (εₜ = Rₜ / ψₜ)")
    st.caption("Esperado: média ≈ 1. εₜ > 1 → range maior que o esperado; εₜ < 1 → menor que o esperado.")
    st.metric("Média de εₜ", f"{df['epsilon'].mean():.4f}")

    fig_eps = go.Figure()
    fig_eps.add_trace(go.Scatter(x=df.index, y=df["epsilon"], mode="markers", name="epsilon",
                                  marker=dict(size=4, color="gray")))
    for y_val, color in [(1, "gray"), (0.5, "red"), (1.5, "red")]:
        fig_eps.add_hline(y=y_val, line=dict(color=color, dash="dash"))
    fig_eps.update_layout(template="plotly_white", height=350, xaxis_title="Data", yaxis_title="epsilon")
    st.plotly_chart(fig_eps, use_container_width=True)

# --- Aba 3: Modelo Adaptativo ---
ADAPTATIVO_MD = r"""
## Como funciona o Modelo Adaptativo

O CARR estimado com a **amostra inteira** (aba anterior) assume que $\omega$, $\alpha$
e $\beta$ são **constantes ao longo de todo o período**. Na prática, o comportamento
do range muda com o tempo — dias de bull market, crises ou lateralização de longo
prazo têm dinâmicas de amplitude bem diferentes. O modelo adaptativo resolve isso
reestimando os parâmetros periodicamente, usando apenas dados recentes:

1. **Janela móvel de WINDOW_PLACEHOLDER dias** — a cada passo, o modelo enxerga somente
   os últimos WINDOW_PLACEHOLDER valores de range para calcular $\psi_t$, "esquecendo"
   deliberadamente o passado distante.
2. **Recalibragem no início de cada mês** — em vez de reotimizar $(\omega,\alpha,\beta)$
   todo dia (caro computacionalmente e instável), a reestimação por máxima
   verossimilhança só ocorre quando o mês da data muda em relação ao dia anterior.
   Nos demais dias, os parâmetros vigentes do último mês continuam sendo usados.
3. **Trava de segurança (estacionariedade)** — uma nova recalibragem só é aceita se o
   otimizador convergiu (`result.success`) **e** se $\alpha+\beta < 0,999$ continuar
   valendo. Caso contrário, o modelo mantém os parâmetros anteriores, evitando que uma
   otimização mal-condicionada faça o CARR "explodir".

O resultado é uma série $\psi_t^{\text{adap}}$ que se adapta a mudanças de regime de
volatilidade — ao custo de ser um pouco mais ruidosa, já que reflete apenas uma
janela curta de dados.
""".replace("WINDOW_PLACEHOLDER", str(window))

with tabs[3]:
    st.markdown(ADAPTATIVO_MD)
    st.divider()
    st.subheader(f"Modelo Adaptativo — recalibragem mensal, janela de {window} dias")

    forecasts, psi_rolling, params_history, recalib_flags = cached_adaptive(
        tuple(df.index.astype(str)), tuple(R.values), tuple(initial_params), window
    )

    df["carr_adap"] = np.nan
    df.loc[df.index[window:], "carr_adap"] = forecasts

    df["psi_adap"] = np.nan
    df.loc[df.index[window:], "psi_adap"] = psi_rolling
    df["z"] = df["range"] / df["psi_adap"]

    params_arr = np.array(params_history)
    df["omega_adap"] = np.nan
    df["alpha_adap"] = np.nan
    df["beta_adap"] = np.nan
    df.loc[df.index[window:], "omega_adap"] = params_arr[:, 0]
    df.loc[df.index[window:], "alpha_adap"] = params_arr[:, 1]
    df.loc[df.index[window:], "beta_adap"] = params_arr[:, 2]

    recalib_dates = df.index[window:][np.array(recalib_flags)]
    n_recalibs = len(recalib_dates)
    omega_now, alpha_now, beta_now = params_history[-1]
    dias_desde_recalib = (df.index[-1] - recalib_dates[-1]).days if n_recalibs else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recalibragens realizadas", n_recalibs)
    c2.metric("ω / α / β vigentes", f"{omega_now:.3f} / {alpha_now:.3f} / {beta_now:.3f}")
    c3.metric("α + β vigente", f"{alpha_now + beta_now:.3f}")
    c4.metric("Dias desde a última recalibragem",
              dias_desde_recalib if dias_desde_recalib is not None else "n/d")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["range"], name="Range", line=dict(color="gray", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["carr_adap"].shift(1), name="CARR Adaptativo", line=dict(color="red", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["carr_est"].shift(1), name="CARR Full-Sample (estático)", line=dict(color="blue", width=1.5)))
    fig.update_layout(
        title="Comparação: Range vs CARR Adaptativo vs CARR Full-Sample",
        xaxis_title="Data", yaxis_title="Range",
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evolução dos parâmetros recalibrados")
    st.caption("Cada 'degrau' representa uma recalibragem mensal aceita pelo modelo.")
    fig_params = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("ω (omega)", "α (alpha)", "β (beta)"),
                                vertical_spacing=0.06)
    fig_params.add_trace(go.Scatter(x=df.index, y=df["omega_adap"], line_shape="hv",
                                     line=dict(color="#9b59b6"), name="omega"), row=1, col=1)
    fig_params.add_trace(go.Scatter(x=df.index, y=df["alpha_adap"], line_shape="hv",
                                     line=dict(color="#3498db"), name="alpha"), row=2, col=1)
    fig_params.add_trace(go.Scatter(x=df.index, y=df["beta_adap"], line_shape="hv",
                                     line=dict(color="#e67e22"), name="beta"), row=3, col=1)
    fig_params.update_layout(template="plotly_white", height=550, showlegend=False)
    st.plotly_chart(fig_params, use_container_width=True)

    st.subheader("Full-Sample vs Adaptativo: qual explica melhor o range recente?")
    st.caption(
        "Erros calculados no período em que ambos os modelos têm previsão "
        "(a partir do fim da primeira janela móvel), comparando a previsão "
        "de cada modelo com o range efetivamente observado no dia seguinte."
    )

    comp = df.iloc[window:].copy()
    err_full = comp["range"] - comp["carr_est"].shift(1)
    err_adap = comp["range"] - comp["carr_adap"].shift(1)

    comparacao = pd.DataFrame({
        "Modelo": ["CARR Full-Sample (estático)", "CARR Adaptativo"],
        "MAE": [err_full.abs().mean(), err_adap.abs().mean()],
        "RMSE": [np.sqrt((err_full**2).mean()), np.sqrt((err_adap**2).mean())],
        "Viés médio (erro médio)": [err_full.mean(), err_adap.mean()],
    }).set_index("Modelo").round(4)
    st.dataframe(comparacao, use_container_width=True)

    melhor = comparacao["RMSE"].idxmin()
    st.info(f"📌 Menor RMSE no período analisado: **{melhor}**.")

    with st.expander("📋 Datas de recalibragem e parâmetros estimados"):
        tabela_recalib = pd.DataFrame({
            "Data": recalib_dates,
        })
        tabela_recalib["omega"] = df.loc[recalib_dates, "omega_adap"].round(4).values
        tabela_recalib["alpha"] = df.loc[recalib_dates, "alpha_adap"].round(4).values
        tabela_recalib["beta"] = df.loc[recalib_dates, "beta_adap"].round(4).values
        tabela_recalib["alpha + beta"] = (tabela_recalib["alpha"] + tabela_recalib["beta"]).round(4)
        st.dataframe(tabela_recalib.set_index("Data"), use_container_width=True)


REGIMES_MD = r"""
## Extensão do Modelo: CARR + Regimes de Mercado

A ideia central consiste em isolar o choque padronizado de volatilidade ($Z_t$),
definindo-o como a razão entre a amplitude observada e a amplitude condicional
estimada pelo modelo CARR:

$$ Z_t = \frac{R_t}{\psi_t} $$

Ao cruzar a magnitude deste resíduo ($Z_t$, ou a própria amplitude $R_t$) com a
direção dos **retornos diários**, podemos classificar o mercado em diferentes
estados (regimes). A matriz abaixo ilustra a categorização bidimensional usada
neste dashboard:

| Retorno Diário | Volatilidade / Range ($Z_t$) | Classificação do Regime | Dinâmica do Mercado |
| :---: | :---: | :--- | :--- |
| **Positivo** ($\uparrow$) | **Baixo** ($\downarrow$) | Tendência Calma | Crescimento sustentável com baixo ruído |
| **Positivo** ($\uparrow$) | **Alto** ($\uparrow$) | Expansão | Movimento de forte tração compradora |
| **Negativo** ($\downarrow$) | **Alto** ($\uparrow$) | Estresse | Pânico, *sell-off* ou correção abrupta |
| **Negativo** ($\downarrow$) | **Baixo** ($\downarrow$) | Contração | Realização de lucros ou exaustão lenta |

> **Nota:** Essa estrutura permite não apenas analisar o risco, mas também construir
> estratégias quantitativas condicionadas ao estado atual do mercado (ex.: alocar
> mais risco em "Tendência Calma" e proteger o portfólio em "Estresse"). Dias em que
> $Z_t$ fica entre os dois limiares (nem alto, nem baixo) ficam como **Neutro**.
"""

REGIME_COLORS = {
    "Tendência Calma": "#2ecc71",   # verde
    "Expansão": "#3498db",          # azul
    "Estresse": "#e74c3c",          # vermelho
    "Contração": "#f39c12",         # laranja
    "Neutro": "#7f8c8d",            # cinza
}

# --- Aba 4: Regimes de Mercado ---
with tabs[4]:
    st.markdown(REGIMES_MD)
    st.divider()
    st.subheader(f"Classificação de regimes — {ticker}")

    positivo = df["return"] >= 0
    z_alto = df["z"] > z_long
    z_baixo = df["z"] < z_short

    condicoes = [
        positivo & z_baixo,
        positivo & z_alto,
        ~positivo & z_alto,
        ~positivo & z_baixo,
    ]
    nomes = ["Tendência Calma", "Expansão", "Estresse", "Contração"]

    df["regime"] = np.select(condicoes, nomes, default="Neutro")
    df.loc[df["z"].isna(), "regime"] = np.nan
    df["return_fwd"] = df["return"].shift(-1)
    df["return_fwd_abs"] = df["return_fwd"].abs()
    # Soma dos retornos dos PRÓXIMOS 5 dias (não inclui o dia atual)
    df["return_fwd_5d"] = (
        df["return"][::-1].rolling(window=5, min_periods=5).sum()[::-1].shift(-1)
    )
    df["return_fwd_5d_abs"] = df["return_fwd_5d"].abs()

    contagem = df["regime"].value_counts()
    total_classificado = contagem.sum()

    cols = st.columns(len(nomes) + 1)
    for col, nome in zip(cols, nomes + ["Neutro"]):
        qtd = int(contagem.get(nome, 0))
        pct = (qtd / total_classificado * 100) if total_classificado else 0.0
        col.metric(nome, f"{qtd} dias", f"{pct:.1f}%")

    with st.expander("📋 Estatísticas por regime (retorno médio, z médio)", expanded= True):
        resumo = (
            df.dropna(subset=["regime"])
            .groupby("regime")
            .agg(
                dias=("regime", "count"),
                retorno_medio=("return_fwd_abs", "mean"),
                mov_5d=("return_fwd_5d_abs", "mean"),
                z_medio=("z", "mean"),
                vol_media=("vol", "mean"),
            )
            .reindex(nomes + ["Neutro"])
            .dropna(how="all")
        )
        resumo["retorno_medio"] = (resumo["retorno_medio"] * 100).round(3)
        resumo["mov_5d"] = (resumo["mov_5d"] * 100).round(3)
        resumo["z_medio"] = resumo["z_medio"].round(3)
        resumo["vol_media"] = (resumo["vol_media"] * 100).round(2)
        resumo = resumo.rename(columns={
            "dias": "Dias",
            "retorno_medio": "Retorno médio absoluto (%) - Dia Posterior",
            "mov_5d": "Movimentação abs. méd. próx. 5 dias (%)",
            "z_medio": "z médio",
            "vol_media": "Vol. anualizada média (%)",
        })
        st.dataframe(resumo, use_container_width=True)
        

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.05,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker,
                              line=dict(color="rgba(255,255,255,0.5)", width=1)), row=1, col=1)

    for nome in nomes:
        mask = df["regime"] == nome
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.index, y=np.where(mask, df["Close"], np.nan),
                mode="markers", name=nome,
                marker=dict(color=REGIME_COLORS[nome], size=6),
            ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["z"], mode="lines", name="z",
                              line=dict(color="gray", width=1)), row=2, col=1)
    fig.add_hline(y=z_long, line=dict(color=REGIME_COLORS["Expansão"], dash="dash"),
                  annotation_text=f"z alto ({z_long})", row=2, col=1)
    fig.add_hline(y=z_short, line=dict(color=REGIME_COLORS["Contração"], dash="dash"),
                  annotation_text=f"z baixo ({z_short})", row=2, col=1)

    fig.update_layout(
        title=f"CARR + regimes de mercado aplicado ao {ticker}",
        template="plotly_dark", height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="z", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    fig_bar = px.bar(
        contagem.reindex(nomes + ["Neutro"]).dropna(),
        orientation="h",
        color=contagem.reindex(nomes + ["Neutro"]).dropna().index,
        color_discrete_map=REGIME_COLORS,
        labels={"value": "Nº de dias", "index": "Regime"},
        title="Distribuição de dias por regime",
    )
    fig_bar.update_layout(template="plotly_dark", showlegend=False, height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Aba 5: VaR ---
with tabs[5]:
    st.subheader("VaR Dinâmico via CARR (Parkinson)")

    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    parkinson_const = np.sqrt(4 * np.log(2))
    df["cond_vol"] = (df["carr_adap"] / 100) / parkinson_const

    z_95 = stats.norm.ppf(0.05)
    z_99 = stats.norm.ppf(0.01)
    df["VaR_95"] = z_95 * df["cond_vol"]
    df["VaR_99"] = z_99 * df["cond_vol"]

    df_valid = df.dropna(subset=["Return", "VaR_95"]).copy()

    breaches_95 = df_valid["Return"] < df_valid["VaR_95"]
    breaches_99 = df_valid["Return"] < df_valid["VaR_99"]
    hit_rate_95 = breaches_95.mean() * 100
    hit_rate_99 = breaches_99.mean() * 100

    c1, c2 = st.columns(2)
    c1.metric("VaR 95% — realizado", f"{hit_rate_95:.2f}%", help="Esperado: 5.00%")
    c1.caption(f"{int(breaches_95.sum())} dias de violação")
    c2.metric("VaR 99% — realizado", f"{hit_rate_99:.2f}%", help="Esperado: 1.00%")
    c2.caption(f"{int(breaches_99.sum())} dias de violação")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid["Return"], mode="lines",
                              name="Retornos", line=dict(color="gray", width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=df_valid.index[breaches_95], y=df_valid["Return"][breaches_95],
                              mode="markers", name="Violação VaR 95%", marker=dict(color="red", size=6)))
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid["VaR_95"], mode="lines",
                              name="VaR 95% (CARR)", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid["VaR_99"], mode="lines",
                              name="VaR 99% (CARR)", line=dict(color="red", width=1.5, dash="dash")))
    fig.update_layout(
        title="Backtesting de VaR Dinâmico via Modelo CARR",
        template="plotly_dark", height=500,
        xaxis_title="Data", yaxis_title="Retorno Logarítmico",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_hist = px.histogram(df, x="VaR_95", nbins=50, title="Distribuição da VaR — CARR Adaptativo")
    fig_hist.update_layout(
        xaxis_title="VaR 95%", yaxis_title="Frequência",
        template="plotly_dark", height=450,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Aba 6: Monte Carlo ---
with tabs[6]:
    st.subheader("VaR via Simulação de Monte Carlo")

    sigma_projetado = df["cond_vol"].iloc[-1]
    np.random.seed(42)
    retornos_simulados = np.random.normal(loc=0.0, scale=sigma_projetado, size=int(n_sims))
    pnl_simulado = capital * retornos_simulados

    var_90 = np.percentile(pnl_simulado, 10)
    var_95 = np.percentile(pnl_simulado, 5)
    var_99 = np.percentile(pnl_simulado, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capital exposto", f"R$ {capital:,.0f}".replace(",", "."))
    c2.metric("Vol. diária (CARR)", f"{sigma_projetado:.2%}")
    c3.metric("VaR 95% (1 dia)", f"R$ {abs(var_95):,.0f}".replace(",", "."))
    c4.metric("VaR 99% (1 dia)", f"R$ {abs(var_99):,.0f}".replace(",", "."))

    st.caption(f"VaR 90% (1 dia): R$ {abs(var_90):,.2f}".replace(",", "."))

    # Histograma calculado manualmente para poder colorir a cauda esquerda (VaR 95%)
    counts, bin_edges = np.histogram(pnl_simulado, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Barras à esquerda do VaR 95% (os piores 5% de cenários) em vermelho
    bar_colors = np.where(bin_centers <= var_95, "crimson", "lightgray")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts, width=bin_width,
        marker_color=bar_colors,
        name="Cenários simulados",
        hovertemplate="PnL: R$ %{x:,.0f}<br>Frequência: %{y}<extra></extra>",
    ))
    fig.add_vline(x=var_95, line=dict(color="darkred", dash="dash", width=2),
                  annotation_text=f"VaR 95% (-R$ {abs(var_95):,.0f})")
    fig.add_vline(x=var_99, line=dict(color="orange", dash="dash", width=2),
                  annotation_text=f"VaR 99% (-R$ {abs(var_99):,.0f})")
    fig.update_layout(
        title=f"Distribuição de Lucros e Prejuízos Simulados (Monte Carlo, {int(n_sims):,} cenários)".replace(",", "."),
        xaxis_title="PnL simulado (R$)", yaxis_title="Frequência",
        template="plotly_white", height=500, showlegend=False,
        bargap=0.02,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Em vermelho: os 5% piores cenários simulados (cauda esquerda, além do VaR 95%).")
