import pandas as pd
import numpy as np
from scipy.stats import linregress
import yfinance as yf # Necessitaràs instal·lar-la: pip install yfinance

# =========================================================================
# === 1. FUNCIÓ DE CÀLCUL DE BETA ASIMÈTRICA ==============================
# =========================================================================

def calcular_betes_asimetriques(df_retorns, nom_rei='BTC-USD', nom_cavaller='DOGE-USD'):
    """
    Calcula la Beta a l'Alça (β+) i la Beta a la Baixa (β-) d'un Cavaller.
    
    Args:
        df_retorns (pd.DataFrame): DataFrame amb els retorns logarítmics (Log Returns).
        nom_rei (str): Ticker del Rei (columna de referència).
        nom_cavaller (str): Ticker del Cavaller (columna a analitzar).

    Returns:
        dict: Diccionari amb els valors de Beta Positiva i Beta Negativa.
    """

    R_R = df_retorns[nom_rei]
    R_C = df_retorns[nom_cavaller]

    # --- Càlcul de la Downside Beta (β-): El Rei BAIXA (R_R < 0) ---
    df_down = df_retorns[R_R < 0]
    
    if len(df_down) > 2:
        slope_down, _, _, _, _ = linregress(df_down[nom_rei], df_down[nom_cavaller])
        beta_down = slope_down
    else:
        beta_down = np.nan

    # --- Càlcul de la Upside Beta (β+): El Rei PUJA (R_R > 0) ---
    df_up = df_retorns[R_R > 0]
    
    if len(df_up) > 2:
        slope_up, _, _, _, _ = linregress(df_up[nom_rei], df_up[nom_cavaller])
        beta_up = slope_up
    else:
        beta_up = np.nan

    return {
        'Beta_Upside_+': beta_up,
        'Beta_Downside_-': beta_down
    }

# =========================================================================
# === 2. DESCARREGAR DADES I APLICAR EL CÀLCUL ============================
# =========================================================================

# Defineix els tickers de Yahoo Finance
TICKER_REI = 'BTC-USD'
TICKER_CAVALLER_BLANC = 'BNB-USD'
TICKER_CAVALLER_REIAL = 'DOGE-USD'
TICKER_CAVALLER_REIAL2 = 'ETH-USD'
TICKER_CAVALLER_REIAL3 = 'SOL-USD'
TICKER_CAVALLER_REIAL4 = 'ADA-USD'

LLISTA_CAVALLERS = [TICKER_CAVALLER_REIAL,TICKER_CAVALLER_REIAL2,TICKER_CAVALLER_REIAL3,TICKER_CAVALLER_REIAL4,TICKER_CAVALLER_BLANC]

# Defineix el període d'anàlisi (període recomanat: 90 dies fins avui)
PERIODE = '90d'
INTERVAL = '1d'

# Llista de tots els actius
tickers = [TICKER_REI, TICKER_CAVALLER_REIAL, TICKER_CAVALLER_BLANC,TICKER_CAVALLER_REIAL2,TICKER_CAVALLER_REIAL3,TICKER_CAVALLER_REIAL4]

print(f"Descarregant dades de {PERIODE} per: {tickers}...")


    # Descàrrega les dades de preus de tancament
df_preus = yf.download(tickers, period=PERIODE, interval=INTERVAL)['Close']
df_preus = df_preus.dropna() # Neteja qualsevol dia amb dades incompletes

    # Càlcul dels Retorns Logarítmics
df_retorns = np.log(df_preus / df_preus.shift(1)).dropna()
    
print(f"Dades utilitzades: {len(df_retorns)} dies.")
    # Diccionari per emmagatzemar els resultats (clau: Ticker, valor: {Betes})
resultats_betes = {}

print("Iniciant càlcul de Betes Asimètriques per a tots els Cavallers...")

# El bucle itera sobre cada Ticker de la llista
for cavaller_ticker in LLISTA_CAVALLERS:
    
    # 1. Apliquem la funció a cada cavaller
    resultat = calcular_betes_asimetriques(
        df_retorns, 
        nom_rei=TICKER_REI, 
        nom_cavaller=cavaller_ticker # El ticker canvia a cada iteració
    )
    
    # 2. Emmagatzemem el resultat al diccionari
    resultats_betes[cavaller_ticker] = resultat
    
    # =========================================================================
    # === 3. RESULTATS I CONCLUSIÓ ============================================
    # =========================================================================

# El segon bucle itera sobre el diccionari de resultats
for ticker, betes in resultats_betes.items():
    print(f"\nCavaller: {ticker}")
    print(f"  Beta a l'Alça (β+): Puja un {betes['Beta_Upside_+']:.2f} % per cada 1% que puja el Rei.")
    print(f"  Beta a la Baixa (β-): Baixa un {betes['Beta_Downside_-']:.2f} % per cada 1% que baixa el Rei.")
