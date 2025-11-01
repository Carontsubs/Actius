import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator 
from google import genai
from google.genai import types
import os # Importem el mòdul os per accedir a les variables d'entorn
from dotenv import load_dotenv # 👈 Importem la funció per carregar .env
import requests

# Configuració de Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Carrega les variables d'entorn del fitxer .env
load_dotenv() 

# Obtenim la clau d'API. Si no la troba, generarà un error.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOKEN_TELEGRAM = os.getenv("TOKEN_TELEGRAM")
CHAT_ID =  os.getenv("CHAT_ID")

# ----------------------------------------------------------------------
# --- FUNCIONS DE CÀLCUL ---
# ----------------------------------------------------------------------

def min_max_scale_log(series):
    """Normalitza una sèrie de dades al rang 1-100 per al càlcul logarítmic.
    AVÍS: Aquesta normalització depèn de la finestra temporal de les dades (repintat).
    """
    min_val, max_val = series.min().item(), series.max().item()
    if max_val == min_val: return pd.Series(50.0, index=series.index)
    return 1 + 99 * (series - min_val) / (max_val - min_val) 

def calculate_obv(df):
    """Calcula l'indicador On-Balance Volume (OBV) de forma vectoritzada."""
    
    # Utilitzem np.sign() per determinar la direcció del canvi de preu (-1, 0, 1)
    price_change = df['Close'].diff().fillna(0)
    direction = np.sign(price_change)
    
    # L'OBV és la suma acumulada de (Volum * Direcció)
    obv_series = (df['Volume'] * direction).cumsum()
    
    # El primer valor d'OBV és sempre 0, així que omplim el NaN creat per .diff()
    # Amb l'OBV del primer tancament (que és 0)
    obv_series = obv_series.fillna(0)
    
    return obv_series

def dades_diaries(df):
    
    # Si les dades no són un MultiIndex (el cas de df_raw), no fem el droplevel
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[:-1]      
    df = df.copy() # Correcció per evitar el SettingWithCopyWarning
    
    # ----------------------------------------------------------------------
    # --- CÀLCULS DEL SISTEMA 1: VOLATILITAT (ATR / V-ATR) ---
    # ----------------------------------------------------------------------

    df['Prev Close'] = df['Close'].shift(1)
    df['Price_TR'] = abs(df['Close'] - df['Prev Close'])
    df['Prev Volume'] = df['Volume'].shift(1)
    df['Volume_VTR'] = abs(df['Volume'] / df['Prev Volume'])
    df['Price_TR_day'] = abs(df['High'] / df['Low'])
    df.dropna(subset=['Price_TR', 'Volume_VTR','Price_TR_day'], inplace=True)

    # 1. EMAs de Volatilitat
    df['TR_EMA'] = df['Price_TR'].ewm(span=21, adjust=False).mean()
    df['VTR_EMA'] = df['Volume_VTR'].ewm(span=21, adjust=False).mean()
    df['TR_EMA13_day'] = df['Price_TR_day'].ewm(span=13, adjust=False).mean()

    # 2. Normalització i Ràtio Logarítmica
    df['TR_Norm_EMA'] = min_max_scale_log(df['TR_EMA'])
    df['VTR_Norm_EMA'] = min_max_scale_log(df['VTR_EMA'])
    MIN_SMOOTHING_FACTOR = 0.0001
    denominator_atr = np.maximum(df['VTR_Norm_EMA'], MIN_SMOOTHING_FACTOR) 
    df['Log_Volatility_Ratio'] = np.log( denominator_atr / df['TR_Norm_EMA'])
    df['Prev_LVR'] = df['Log_Volatility_Ratio'].shift(1)
    df['m_LVR'] = df['Log_Volatility_Ratio'] - df['Prev_LVR']

    # ----------------------------------------------------------------------
    # --- CÀLCULS DEL SISTEMA 2: TENDÈNCIA / PRESSIÓ (Preu / OBV) ---
    # ----------------------------------------------------------------------

    df['OBV'] = calculate_obv(df)
    df['OBV_EMA'] = df['OBV'].ewm(span=21, adjust=False).mean()

    df['Close_EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['Close_EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['Close_EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['Close_EMA233'] = df['Close'].ewm(span=233, adjust=False).mean()

    # 3. Normalització i Ràtio Logarítmica
    df['Close_EMA_Norm'] = min_max_scale_log(df['Close_EMA21'])
    df['OBV_EMA_Norm'] = min_max_scale_log(df['OBV_EMA'])   
    denominator_obv = df['OBV_EMA_Norm'].replace(0, 1e-9)
    df['Log_Divergence_Ratio'] = np.log( denominator_obv / df['Close_EMA_Norm'])
    df['Prev_LDR'] = df['Log_Divergence_Ratio'].shift(1)
    df['m_LDR'] = df['Log_Divergence_Ratio'] - df['Prev_LDR']


    df['RED'] = abs(df['Log_Divergence_Ratio']) / abs(df['Log_Volatility_Ratio'])
    

    df['Prev_RED'] = df['RED'].shift(1)
    df['m_RED'] = df['RED'] - df['Prev_RED']


    # Neteja final de NaNs introduïts per les EMAs
    df.dropna(inplace=True) 

    # -----------------------------------------------------------
    # FUNCIONS DE CÀLCUL D'INDICADORS (ADX, ATR, RSI)
    # -----------------------------------------------------------

    def calculate_adx(df, period=14):
        """Calcula l'Average Directional Index (ADX), +DI i -DI. (ADX manual)"""
        
        df_adx = df.copy()

        # 1. True Range (TR)
        df_adx['H-L'] = df_adx['High'] - df_adx['Low']
        df_adx['H-PC'] = np.abs(df_adx['High'] - df_adx['Close'].shift(1))
        df_adx['L-PC'] = np.abs(df_adx['Low'] - df_adx['Close'].shift(1))
        df_adx['TR'] = df_adx[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # 2. Directional Movement (+DM i -DM)
        df_adx['+DM'] = np.where(
            (df_adx['High'] - df_adx['High'].shift(1) > 0) & 
            (df_adx['High'] - df_adx['High'].shift(1) > df_adx['Low'].shift(1) - df_adx['Low']), 
            df_adx['High'] - df_adx['High'].shift(1), 
            0
        )
        df_adx['-DM'] = np.where(
            (df_adx['Low'].shift(1) - df_adx['Low'] > 0) & 
            (df_adx['Low'].shift(1) - df_adx['Low'] > df_adx['High'] - df_adx['High'].shift(1)), 
            df_adx['Low'].shift(1) - df_adx['Low'], 
            0
        )

        # 3. ATR, +DI i -DI (Wilder's Smoothing)
        def wilder_smooth(series, period):
            return series.ewm(alpha=1/period, adjust=False).mean()
        
        df_adx['ATR_ADX'] = wilder_smooth(df_adx['TR'], period)
        df_adx['+DI'] = 100 * (wilder_smooth(df_adx['+DM'], period) / df_adx['ATR_ADX'])
        df_adx['-DI'] = 100 * (wilder_smooth(df_adx['-DM'], period) / df_adx['ATR_ADX'])

        # 4. Directional Index (DX)
        # Gestionar la divisió per zero si +DI + -DI és 0
        sum_di = df_adx['+DI'] + df_adx['-DI']
        df_adx['DX'] = np.where(sum_di > 0, 100 * (np.abs(df_adx['+DI'] - df_adx['-DI']) / sum_di), 0)

        # 5. Average Directional Index (ADX)
        df_adx['ADX'] = wilder_smooth(df_adx['DX'], period)

        return df_adx[['+DI', '-DI', 'ADX']]


    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=21).average_true_range()
    df['ATR_Q5'] = df['ATR'].rolling(window=55).quantile(0.05)
    df['ATR_Q90'] = df['ATR'].rolling(window=55).quantile(0.90)
    
    df['RSI'] = RSIIndicator(close=df['Close'], window=21).rsi()

    # Càlculs de Volum i Ràtios (REPV)
    df['EMA13_Close'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['SMA_55_Volume'] = df['Volume'].rolling(window=55).mean()
    df['SMA_13_Volume'] = df['Volume'].rolling(window=21).mean()
    df['Price_TR_ema8'] = df['Price_TR'].ewm(span=5, adjust=False).mean()
    df['Volume_ema8'] = df['Volume'].ewm(span=5, adjust=False).mean()

    # Gestionar la divisió per zero
    denominator_sma = df['SMA_13_Volume'].replace(0, 1e-9)
    df['REPV'] =  denominator_sma / df['ATR'] 
    denominator_vol_ema = df['Volume_ema8'].replace(0, 1e-9)
    df['REPV_a'] = denominator_vol_ema / df['Price_TR_ema8']
    denominator_repv = df['REPV'].replace(0, 1e-9)
    df['REPV_R'] = df['REPV_a'] / denominator_repv

    df['IPE'] = df['Log_Divergence_Ratio'] / df['REPV_R']

    # OSCIL·LADOR ESTOCÀSTIC (Preu)
    period = 21
    smooth_k = 1
    smooth_d = 3

    df['Low_8'] = df['Low'].rolling(window=period).min()
    df['High_8'] = df['High'].rolling(window=period).max()
    denominator_stoch = (df['High_8'] - df['Low_8']).replace(0, 1e-9)
    df['Fast_%K'] = 100 * ((df['Close'] - df['Low_8']) / denominator_stoch)
    df['Slow_%K'] = df['Fast_%K'].rolling(window=smooth_k).mean()
    df['Slow_%D'] = df['Slow_%K'].rolling(window=smooth_d).mean()

    # OSCIL·LADOR RED - ESTOCÀSTIC
    period_vtr = 21 
    smooth_k_vtr = 3
    smooth_d_vtr = 3

    df['Low_VTR'] = df['VTR_EMA'].rolling(window=period_vtr).min()
    df['High_VTR'] = df['VTR_EMA'].rolling(window=period_vtr).max()
    
    # Gestionar la divisió per zero si High_RED - Low_RED és 0
    red_range = df['High_VTR'] - df['Low_VTR']
    df['Fast_%VTR-K'] = np.where(red_range > 0, 
                                 100 * ((df['VTR_EMA'] - df['Low_VTR']) / red_range), 
                                 50) # Valor per defecte al 50 si no hi ha rang

    df['Slow_%VTR-K'] = df['Fast_%VTR-K'].rolling(window=smooth_k_vtr).mean()
    df['Slow_%VTR-D'] = df['Slow_%VTR-K'].rolling(window=smooth_d_vtr).mean()
    
    # OSCIL·LADOR ATR - ESTOCÀSTIC
    period_red = 21 
    smooth_k_red = 3    
    smooth_d_red = 3

    df['Low_ATR'] = df['ATR'].rolling(window=period_red).min()
    df['High_ATR'] = df['ATR'].rolling(window=period_red).max()
    
    # Gestionar la divisió per zero si High_RED - Low_RED és 0
    red_range = df['High_ATR'] - df['Low_ATR']
    df['Fast_%ATR-K'] = np.where(red_range > 0, 
                                 100 * ((df['ATR'] - df['Low_ATR']) / red_range), 
                                 50) # Valor per defecte al 50 si no hi ha rang

    df['Slow_%ATR-K'] = df['Fast_%ATR-K'].rolling(window=smooth_k_red).mean()
    df['Slow_%ATR-D'] = df['Slow_%ATR-K'].rolling(window=smooth_d_red).mean()


    # Afegir l'ADX calculat al DataFrame principal
    df = df.join(calculate_adx(df), how='left')
    
    # # Neteja final de NaNs
    # df.dropna(inplace=True) 
    # RED_f = df['RED'].copy()
    # RED_f[RED_f > 10] = 10

    return df

def envia_missatge(text):
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": text}
    resposta = requests.get(url, params=params)
    print(f"Estat de l'enviament a Telegram: {resposta.json()}")

# 1. Extreure les dades de Yahoo Finance
ticker = "BTC-USD"

# Descarregar les dades històriques
df = yf.download(ticker, period="2y", interval='1d')
df_raw_1 = yf.download(ticker, period="3mo", interval='1h')
df_raw = yf.download(ticker, period="3mo", interval='4h')

df = dades_diaries(df)
df_raw = dades_diaries(df_raw)
df_raw_1 = dades_diaries(df_raw_1)

# ----------------------------------------------------------------------
# --- CÀLCUL DE LLINDARS DINÀMICS (ROLLING QUANTILE) ---
# ----------------------------------------------------------------------
# def llindars(df_raw,interval):

#     if interval is interval1:
#         WINDOW = 250
#     elif interval is interval2:
#         WINDOW = 288
#     else:
#         WINDOW = 72
    # 4h (df_raw) - Finestra de 72 períodes (aprox. 12 dies, 6 candeles/dia * 12 dies)
WINDOW_4H = 72
window = min(WINDOW_4H, len(df_raw))

df_raw['LDR_Q10'] = df_raw['Log_Divergence_Ratio'].rolling(window=window).quantile(0.10)
df_raw['LDR_Q90'] = df_raw['Log_Divergence_Ratio'].rolling(window=window).quantile(0.90)
df_raw['LVR_Q10'] = df_raw['Log_Volatility_Ratio'].rolling(window=window).quantile(0.10)
df_raw['LVR_Q90'] = df_raw['Log_Volatility_Ratio'].rolling(window=window).quantile(0.90)
df_raw['REPV_R_Q10'] = df_raw['REPV_R'].rolling(window=window).quantile(0.10)
df_raw['REPV_R_Q90'] = df_raw['REPV_R'].rolling(window=window).quantile(0.90)
df_raw['IPE_Q10'] = df_raw['IPE'].rolling(window=window).quantile(0.10)
df_raw['IPE_Q90'] = df_raw['IPE'].rolling(window=window).quantile(0.90)

# Diari (df) - Finestra de 250 dies (aprox. 1 any de trading)
WINDOW_DAILY = 250
# Assegurem que la finestra no sigui més gran que la mida de les dades
window_daily = min(WINDOW_DAILY, len(df))

df['LDR_Q10'] = df['Log_Divergence_Ratio'].rolling(window=window_daily).quantile(0.10)
df['LDR_Q90'] = df['Log_Divergence_Ratio'].rolling(window=window_daily).quantile(0.90)
df['LVR_Q10'] = df['Log_Volatility_Ratio'].rolling(window=window_daily).quantile(0.10)
df['LVR_Q90'] = df['Log_Volatility_Ratio'].rolling(window=window_daily).quantile(0.90)
df['REPV_R_Q10'] = df['REPV_R'].rolling(window=window_daily).quantile(0.10)
df['REPV_R_Q90'] = df['REPV_R'].rolling(window=window_daily).quantile(0.90)
df['IPE_Q10'] = df['IPE'].rolling(window=window_daily).quantile(0.10)
df['IPE_Q90'] = df['IPE'].rolling(window=window_daily).quantile(0.90)


# 1h (df_raw) - Finestra de 288 períodes (aprox. 12 dies, 24 candeles/dia * 12 dies)
WINDOW_1H = 288
window_1h = min(WINDOW_1H, len(df_raw_1))

# Càlcul dels percentils utilitzant la nova finestra de 288
df_raw_1['LDR_Q10'] = df_raw_1['Log_Divergence_Ratio'].rolling(window=window_1h).quantile(0.10)
df_raw_1['LDR_Q90'] = df_raw_1['Log_Divergence_Ratio'].rolling(window=window_1h).quantile(0.90)
df_raw_1['LVR_Q10'] = df_raw_1['Log_Volatility_Ratio'].rolling(window=window_1h).quantile(0.10)
df_raw_1['LVR_Q90'] = df_raw_1['Log_Volatility_Ratio'].rolling(window=window_1h).quantile(0.90)
df_raw_1['REPV_R_Q10'] = df_raw_1['REPV_R'].rolling(window=window_1h).quantile(0.10)
df_raw_1['REPV_R_Q90'] = df_raw_1['REPV_R'].rolling(window=window_1h).quantile(0.90)
df_raw_1['IPE_Q10'] = df_raw_1['IPE'].rolling(window=window_1h).quantile(0.10)
df_raw_1['IPE_Q90'] = df_raw_1['IPE'].rolling(window=window_1h).quantile(0.90)

# Neteja de NaNs introduïts pels Rolling Windows
df.dropna(subset=['LDR_Q10', 'LDR_Q90'], inplace=True)
df_raw.dropna(subset=['LVR_Q10', 'LVR_Q90', 'REPV_R_Q10', 'REPV_R_Q90'], inplace=True)
df_raw_1.dropna(subset=['LVR_Q10', 'LVR_Q90', 'REPV_R_Q10', 'REPV_R_Q90'], inplace=True)

df = df.tail(90)
df_raw = df_raw.tail(72)
df_raw_1 = df_raw_1.tail(72)


# Inicialitza el client passant la clau directament.
client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------------------------------------------------------------
# Dades Astronòmiques (Aquestes dades es mantenen fixes per a tots els signes)
# ----------------------------------------------------------------------

print(f"Generant informe...")  # --- 3. CONSTRUCCIÓ DEL PROMPT FINAL ---

prompt = f"""
  [ROL I INSTRUCCIONS]
  **ROL:** Ets un trader especialitzat en mercats volatils amb poca liquidtat amb vocacio divulgativa.
  
  Objectiu: Analitzar la situació actual del mercat identificant la tendència, la volatilitat, i el moment/pressió. Utilitza la darrera fila disponible dels tres DataFrames adjunts (Diari, 4h i 1h) per a realitzar una avaluació.

1. Definició i Lògica dels Indicadors Personalitzats
Els meus indicadors principals es basen en la relació logarítmica entre indicadors normalitzats de preu i volum. Les referències de Quantils Dinàmics (Q10/Q90) s'han d'utilitzar com a llindars per identificar condicions extremes (mínims i màxims històrics recents).

Log_Volatility_Ratio (LVR): Mesura la liquiditat. Un LVR Positiu i Alt suggereix una alta liquidtat que ofereix friccio al moviment. Si és Negatiu i Alt en valor absolut, hi ha poca liquiditat fent que els preus es puguin moure amb molt volatilitat.

Log_Divergence_Ratio (LDR): Mesura la cuantitat de volum acumulat. S'ha de mirar la tendencia, si puja en un mercat alcista indica que part de la pujada es s'acumula, i si baixa enn un baixista s'esta retirnar de mercat mes del que hauria.

REPV_R (Impuls de Volum Recent): Aquesta ràtio indica el recent impuls de volum respecte a la volatilitat base. Un Valor Alt indica una forta injecció recent de volum. les mans fortes comaprant o venent,aixos'ha de veure coparnt amb les estocastiques.

ESTOATR: EStocatica del ATR que mediex la volatilitat, entenen una volatilitat baixa com un mercat alcista i una volatilitat alta com un mercat baixista.

ESTOVTR: Estocasitca de la variacio de volum, indicant junstanment amb la estocastica del preu si son acumulacions les dues juntes o distribucions quan divergeixen.

2. Punts Específics d'Anàlisi
Per a l'última candela de cada període (Diari, 4h, 1h), realitza les següents avaluacions:

A. Avaluació de la Tendència i Pressió (LDR, EMA233, ADX)
Tendència Cíclica (Diari i 4h): Avalua la posició del Close actual respecte a la Close_EMA233 en la de 1D, en els marcs de 4H i 1H compara amb la Close_EMA21 per determinar la tendència a llarg termini.

Pressió Extrema: Compara el LDR actual amb els seus llindars de Q10 i Q90. Està en un rang extrem, indicant una pressió que podria precedir un canvi o una forta continuació?

Força Direccional: Indica la tendencia de l ADX

B. Evaluacio del Volum.
Força de Volum: Indica les variacion de volum important sobretot respecte a la sma55. 

Direccio del Volum: Indica si es produeixen en caigudes o pujades.

Abesencia de dades: Informe de la falta de dades en el marc que toqui i com dificulta el analisi.

C. Avaluació de la Volatilitat i l'Impuls (LVR, ATR, REPV_R)
Extrems de Volatilitat: El valor de l'ATR està per sota del seu Q5 (volatilitat mínima) o per sobre del Q90 (volatilitat màxima)? Això suggereix un potencial d'expansió o contracció imminent de la volatilitat.

Qualitat de la Volatilitat: Compara el LVR amb els seus Q10/Q90. Està la volatilitat impulsada pel volum (LVR Alt) o el preu (LVR Baix)?

Impuls de Volum Recent: El REPV_R actual està per sobre del Q90? Indica una recent i forta injecció de volum en el mercat.

D.** Evaluacio de les estoactiques**(Slow %D, Slow %VTR-D i Slow %ATR-D).

Evaular direccions de cada escuna

Possibles creuaments aixi com divergencies, en un futur proper entre la ESTOATR i la ESTO del preu.

3. **Síntesi i Pronòstic**
Proporciona una conclusió sintètica per a cada període (Diari, 4h, 1h). Recotrda que nomes et donc les darreres dades de cada df, pero el calcul es de 2 anys per les 1D, i 3 mesos per les altres dues. La síntesi ha d'incloure:

Tendència General (Alcista, Baixista o Consolidació).

Estat de Volatilitat (Alta, Baixa, o Normal), respecte a les dades de mostra

Pressió Dominant (Acumulació, Distribució, o Neutre).

Situacio de les estocastiques.

Conclusió Operativa: Les dades suggereixen una continuació, una possible reversió o una fase de consolidació/incertesa?


El informe ha de ser de **no mes de 400 paraules** , estructurat en blocs i no utilitzis la negreta.

[DADES A CONTINUACIO]

*** DADES 1D***
{df}

*** DADES DE 4H ***
{df_raw}

*** DADES DE 1H ***
{df_raw_1}
"""



# Definim la configuració per a totes les crides a l'API
configuracio_ia = types.GenerateContentConfig(
    temperature=0.3,        # Equilibri entre coherència i creativitat
    # max_output_tokens=350,  # Mantenir la resposta curta (aprox. 150 paraules)
    top_p=0.9,              # Bon control d'aleatorietat
    top_k=40,               # Limita la selecció a les 40 paraules més probables
    # stop_sequences=['.']  # Opcional: Aturar-se en un punt
)

  # 2. Fes la crida a l'API
try:
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=configuracio_ia)    

  # 3. Emmagatzema el resultat
  # horoscops_generats[signe] = response.text
  print(response.text)
  envia_missatge(response.text)

except Exception as e:
  print(f"❌ ERROR. {e}")

