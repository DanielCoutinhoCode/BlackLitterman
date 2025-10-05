# Tentativa de implementar o modelo Black-Litterman
# V2 conta com limite de peso mínimo e máximo para cada ativo
# V2 conta com cálculo do VaR Percentual
# V3 conta com análise de cenários de estresse e CVar e Var Histórico
# V4 é uma versão de testes

# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import BlackLittermanModel, risk_models, expected_returns, plotting
from pypfopt import EfficientFrontier, DiscreteAllocation
from pypfopt import objective_functions
from pypfopt import black_litterman
from scipy import stats
from pypfopt import EfficientCVaR

# Definindo os ativos
tickers = ["PETR4.SA", "VALE3.SA", "BBAS3.SA", "BRSR6.SA", "TAEE11.SA", 
           "CPLE6.SA", "SAPR3.SA", "CSMG3.SA", "CVCB3.SA"]

# Obter dados históricos de preços
precos = yf.download(tickers, period="10y", auto_adjust=False)["Adj Close"]
precos.ffill(inplace=True)
preco_mercado = yf.download("^BVSP", period="10y", auto_adjust=False)["Adj Close"]
preco_mercado.ffill(inplace=True)

# Calcular matriz de cov

S = risk_models.CovarianceShrinkage(precos).ledoit_wolf()

# Obter capitalização de mercado
market_caps = {}
for t in tickers:
    stock = yf.Ticker(t)
    market_caps[t] = stock.info['marketCap']
mcaps = pd.Series(market_caps)

# Calcular os retornos de equilíbrio implícitos pelo mercado
# delta = coeficiente de aversão ao risco médio do mercado (2.5 é um valor comum)
risk_free_rate = selic = 0.094
delta = black_litterman.market_implied_risk_aversion(preco_mercado, selic)
retornos_totais_implicitos = black_litterman.market_implied_prior_returns(mcaps, delta, S, selic)
pi = retornos_totais_implicitos - selic
print("Retornos de equilíbrio implícitos pelo mercado:")
print(pi * 100, "\n")

# Definir as opiniões
P = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
Q = np.array([0.00])
confidences = [1]

# Executar o BL
bl = BlackLittermanModel(S, pi=pi, P=P, Q=Q, omega="idzorek", view_confidences=confidences)
retornos_posteriores = bl.bl_returns()

print("Retornos Esperados Após combinar com as visões")
print(retornos_posteriores.apply(lambda x: f"{x:.2%}"))
print("-" * 50)

# Otimização do portfólio
# Adicionando peso mínimo e máximo para cada ativo
ef = EfficientFrontier(retornos_posteriores, S, weight_bounds=(0.0, 0.2))
ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Regularização L2
pesos = ef.max_sharpe(risk_free_rate=selic)
pesos_limpos = ef.clean_weights()

print("Alocação da Carteira Ótima (Black-Litterman):")
print(pd.Series(pesos_limpos).apply(lambda x: f"{x:.2%}"))
print("-" * 50)

print("\nDesempenho da Carteira Final")
ef.portfolio_performance(verbose=True, risk_free_rate=selic)

# Cálculo do VaR Paramétrico
print("\n" + "="*50)
print("CÁLCULO DO VALUE AT RISK PARAMÉTRICO (VaR) PERCENTUAL")
print("="*50)

# Cálculo do VaR - Parâmetros do VaR
confianca = 0.99
z_score = stats.norm.ppf(confianca) # Calcula o Z-score exato

# Cálculo do VaR - Calcular a volatilidade DIÁRIA da carteira
# Cálculo do VaR - A matriz 'S' é anualizada, então precisamos convertê-la para diária
S_diaria = S / 252
pesos_array = np.array(list(pesos_limpos.values()))
vol_diaria_portfolio = np.sqrt(np.dot(pesos_array.T, np.dot(S_diaria, pesos_array)))

# Cálculo do VaR - Calcular o VaR Percentual
var_percentual = z_score * vol_diaria_portfolio

# Cálculo do VaR - Apresentar os resultados
print(f"Intervalo de Confiança: {confianca:.0%}")
print(f"Z-Score correspondente (monocaudal): {z_score:.2f}")
print("-" * 50)
print(f"Volatilidade Diária da Carteira: {vol_diaria_portfolio:.4%}")
print(f"VaR (1 dia, {confianca:.0%}) Percentual: {var_percentual:.4%}")
print("-" * 50)

print("\nInterpretação:")
print(f"Com {confianca:.0%} de confiança, a perda máxima esperada para esta carteira em 1 dia é de {var_percentual:.4%}.")

# Cálculo do CVar
def calcular_cvar_historico(precos, pesos, confianca=0.99):
    
    print("\n" + "="*50)
    print(f"CÁLCULO DO VaR E CVaR HISTÓRICO ({confianca:.0%})")
    print("="*50)

    # 1. Preparar os pesos e calcular os retornos diários dos ativos
    pesos_array = np.array(list(pesos.values()))
    retornos_diarios_ativos = precos.pct_change()

    # 2. Calcular a série temporal de retornos diários da carteira
    retornos_diarios_carteira = retornos_diarios_ativos.dot(pesos_array)
    retornos_diarios_carteira = retornos_diarios_carteira.dropna()

    # 3. Calcular o VaR Histórico no quantil especificado
    var_historico = retornos_diarios_carteira.quantile(1 - confianca)

    # 4. Calcular o CVaR Histórico (média dos retornos piores que o VaR)
    cvar_historico = retornos_diarios_carteira[retornos_diarios_carteira <= var_historico].mean()

    # 5. Apresentar os resultados
    print(f"VaR Histórico (1 dia, {confianca:.0%}): {var_historico:.4%}")
    print(f"CVaR Histórico (1 dia, {confianca:.0%}): {cvar_historico:.4%}")
    print("-" * 50)
    
    print("\nInterpretação:")
    print(f"VaR: Com {confianca:.0%} de confiança, a perda diária não deve exceder {var_historico:.4%}.")
    print(f"CVaR: Nos {1-confianca:.0%} piores dias, a perda média diária esperada seria de {abs(cvar_historico):.4%}.")

calcular_cvar_historico(precos, pesos_limpos, confianca=0.99)

# Stress Testing - Simulação de Choques de Retorno

def realizar_stress_test(pesos, cenario_nome, data_inicio_cenario, data_fim_cenario):
    print(f"\n--- Teste de Estresse: {cenario_nome} ---")
    print(f"Período analisado: {data_inicio_cenario} a {data_fim_cenario}")
    tickers_cenario = list(pesos.keys())
    pesos_array = np.array(list(pesos.values()))
    try:
        precos_cenario = yf.download(
            tickers_cenario, 
            start=data_inicio_cenario, 
            end=data_fim_cenario,
            auto_adjust=False,
            progress=False
        )['Adj Close']
        precos_cenario.ffill(inplace=True)
        if precos_cenario.empty:
            print("Não foram encontrados dados para o período especificado.")
            return
        retornos_cenario = (precos_cenario.iloc[-1] / precos_cenario.iloc[0]) - 1
        retorno_carteira_cenario = np.dot(pesos_array, retornos_cenario)
        print(f"Resultado: A carteira teria tido um retorno de {retorno_carteira_cenario:.2%} durante este período.")
    except Exception as e:
        print(f"Não foi possível realizar o teste para este cenário. Erro: {e}")

print("\n" + "="*50)
print("ANÁLISE DE CENÁRIOS DE ESTRESSE")
print("="*50)

realizar_stress_test(pesos_limpos, "Crise da COVID-19", "2020-02-19", "2020-03-23")
realizar_stress_test(pesos_limpos, "Joesley Day", "2017-05-17", "2017-05-19")
realizar_stress_test(pesos_limpos, "Crise Financeira Global de 2008", "2008-09-15", "2008-11-20")
realizar_stress_test(pesos_limpos, "Rally Pós-COVID", "2020-03-24", "2020-12-31")

# # Plot - Retornos de equilíbrio implícitos pelo mercado
# plt.figure(figsize=(10, 4))
# (pi * 100).plot(kind='bar', color='skyblue')
# plt.title("Retornos de equilíbrio implícitos pelo mercado (%)")
# plt.ylabel("Retorno (%)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Plot - Retornos Esperados Após combinar com as visões
# plt.figure(figsize=(10, 4))
# (retornos_posteriores * 100).plot(kind='bar', color='orange')
# plt.title("Retornos Esperados Após combinar com as visões (%)")
# plt.ylabel("Retorno (%)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Plot - Alocação da Carteira Ótima (Black-Litterman)
# plt.figure(figsize=(10, 4))
# pd.Series(pesos_limpos).plot(kind='bar', color='green')
# plt.title("Alocação da Carteira Ótima (Black-Litterman)")
# plt.ylabel("Peso (%)")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
