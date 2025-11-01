import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator 

# Configuració de Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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
    period_red = 21 
    smooth_k_red = 3
    smooth_d_red = 3

    df['Low_RED'] = df['VTR_EMA'].rolling(window=period_red).min()
    df['High_RED'] = df['VTR_EMA'].rolling(window=period_red).max()
    
    # Gestionar la divisió per zero si High_RED - Low_RED és 0
    red_range = df['High_RED'] - df['Low_RED']
    df['Fast_%RED-K'] = np.where(red_range > 0, 
                                 100 * ((df['VTR_EMA'] - df['Low_RED']) / red_range), 
                                 50) # Valor per defecte al 50 si no hi ha rang

    df['Slow_%RED-K'] = df['Fast_%RED-K'].rolling(window=smooth_k_red).mean()
    df['Slow_%RED-D'] = df['Slow_%RED-K'].rolling(window=smooth_d_red).mean()
    
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

def df_net(df):
    df = df[['Close','Close_EMA21','Close_EMA233','Volume','SMA_55_Volume', 
             'ATR', 'RSI','Log_Divergence_Ratio', 'Fast_%RED-K', 'REPV_R','REPV_R_Q10', 'REPV_R_Q90', # Afegits els quantils
             'Log_Volatility_Ratio','LVR_Q10', 'LVR_Q90','RED','+DI','-DI','ADX' ]]
    return df
def df_net_raw(df):
    df = df[['Close','ATR','Volume','SMA_55_Volume','Log_Volatility_Ratio', 'LVR_Q10', 'LVR_Q90',
             'RED','REPV_R','REPV_R_Q10', 'REPV_R_Q90', # Afegits els quantils
             'Fast_%RED-K', 'Slow_%RED-D']]
    return df


nom_fitxer = 'dades_completes_dinamiques.txt'
with open(nom_fitxer, 'w', encoding='utf-8') as f:
    f.write('Dades del darrer tancament (Diari)\n')
    f.write(df_net(df).tail(10).T.to_string())
    f.write('\n\n Dades de les ultimes 24h (1h)\n')
    f.write(df_net_raw(df_raw_1).tail(25).to_string())
    f.write('\n\n Dades de les ultimes 24h (4h)\n')
    f.write(df_net_raw(df_raw).tail(25).to_string())

print(f"Les dades completes s'han exportat a '{nom_fitxer}'.")

# ----------------------------------------------------------------------
# --- VISUALITZACIÓ DE LA MATRIU 2X2 ---
# --- ELS LLINDARS ARA SÓN DINÀMICS (ROLLING QUANTILE) ---
# ----------------------------------------------------------------------

def grafica(df_raw):

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Anàlisi {ticker}", fontsize=10, y=0.98)

    # --- GRÀFIC 1: RÀTIO LOGARÍTMICA (VOLATILITAT) - ÚS DE LLINDARS MÒBILS (df_raw) ---
    ax1 = axes[1, 1]
    ax1.plot(df_raw.index, df_raw['Log_Volatility_Ratio'], color='darkgreen', linewidth=1.5)
    ax1.set_title("4. Fricció (LVR) i Cost (Vol/Preu)", fontsize=8)
    # ÚS DELS ROLLING QUANTILS PER A LVR
    ax1.plot(df_raw.index, df_raw['LVR_Q10'], color='darkgreen', linestyle='--', linewidth=0.8)
    ax1.plot(df_raw.index, df_raw['LVR_Q90'], color='darkgreen', linestyle='--', linewidth=0.8)
    # ax1.set_ylabel("LN(Ràtio)")
    ax1.grid(True, linestyle=':', alpha=0.8)
    ax1.legend(['LVR'], loc='upper left')
    ax1.tick_params(axis='y', labelcolor='darkgreen', labelsize=8)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # --- EIX Y DRETA: REPV_R (Magnitud, el COST) - ÚS DE LLINDARS MÒBILS (df_raw) ---
    ax1_twin = ax1.twinx()  # Creació d'un segon Eix Y
    # ax1_twin.set_ylabel('RF_Norm (Cost d\'Intervenció)', color='red', fontsize=12)
    ax1_twin.plot(df_raw.index, df_raw['REPV_R'], color='red', linewidth=1.5, label='REPV_R (Cost)')
    # ÚS DELS ROLLING QUANTILS PER A REPV_R
    ax1_twin.plot(df_raw.index, df_raw['REPV_R_Q10'], color='red', linestyle='--', linewidth=0.8)
    ax1_twin.plot(df_raw.index, df_raw['REPV_R_Q90'], color='red', linestyle='--', linewidth=0.8)
    ax1_twin.tick_params(axis='y', labelcolor='red', labelsize=8)
    ax1_twin.legend(['Cost'], loc='lower left')

    # --- GRÀFIC 2: COMPONENTS NORMALITZADES (RED STOCHASTIC) ---
    ax2 = axes[0, 1]
    ax2.set_title("2. LDR", fontsize=8)
    # ax2.plot(df_raw.index, df_raw['Fast_%ATR-K'], label='Rapida', color='blue', linewidth=2)
    # ax2.plot(df_raw.index, df_raw['Slow_%ATR-D'], label='Lenta', color='green', linewidth=2)
    ax2.plot(df_raw.index, df_raw['Log_Divergence_Ratio'], color='purple', linewidth=1.5)
    ax2.plot(df_raw.index, df_raw['LDR_Q10'], color='purple', linestyle='--', linewidth=0.8)
    ax2.plot(df_raw.index, df_raw['LDR_Q90'], color='purple', linestyle='--', linewidth=0.8)
    # ax2.set_ylabel('RED', fontsize=12)
    # ax2.axhline(y=20, color='blue', linestyle='--', linewidth=0.5)
    # ax2.axhline(y=80, color='blue', linestyle='--', linewidth=0.5)
    # ax2.legend(loc='upper left')
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)
    ax2.grid(True, linestyle=':', alpha=0.8)

    # ax2_twin = ax2.twinx()  # Creació d'un segon Eix Y
    # ax2_twin.plot(df_raw.index, df_raw['Close_EMA8'], label='EMA 8', color='orange', linewidth=2)
    # ax2_twin.tick_params(axis='y', labelcolor='orange', labelsize=8)
    # ax2_twin.legend(['EMA 8'], loc='lower left')

    # --- GRÀFIC 3: RÀTIO LOGARÍTMICA (DIVERGÈNCIA / OBV) - ÚS DE LLINDARS MÒBILS (df) ---
    ax3 = axes[1, 0]
    # ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    # ÚS DELS ROLLING QUANTILS PER A LDR
    # ax3.plot(df_raw.index, df_raw['VTR_EMA'], color='purple', linewidth=1.5)
    # ax3.plot(df_raw.index, df_raw['Fast_%RED-K'], label='Rapida', color='blue', linewidth=2)
    ax3.plot(df_raw.index, df_raw['Slow_%D'], color='blue', linewidth=2, label='Preu')
    ax3.plot(df_raw.index, df_raw['Slow_%ATR-D'], color='green', linewidth=2, label='ATR')
    ax3.plot(df_raw.index, df_raw['Slow_%RED-D'], color='red', linewidth=2, label='V-ATR')
    ax3.axhline(y=20, color='blue', linestyle='--', linewidth=0.5)
    ax3.axhline(y=80, color='blue', linestyle='--', linewidth=0.5)
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=0.5)
    # ax3.plot(df_raw.index, df_raw['IPE_Q10'], color='purple', linestyle='--', linewidth=0.8)
    # ax3.plot(df_raw.index, df_raw['IPE_Q90'], color='purple', linestyle='--', linewidth=0.8)
    ax3.set_title("3.Estocastics", fontsize=8)
    # ax3.set_ylabel("LN(Ràtio)")
    # ax3.set_xlabel("Data")
    ax3.grid(True, linestyle=':', alpha=0.8)
    ax3.legend(loc='upper left')
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelcolor='purple', labelsize=8)

    # --- GRÀFIC 4: COMPONENTS DE PREU I TENDÈNCIA ---
    ax4 = axes[0, 0]
    ax4.set_ylabel('Preu', fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.8)
    ax4.set_title("1. Preu i EMAs", fontsize=8)
    ax4.plot(df_raw.index, df_raw['Close'], label='Preu', color='black', alpha=0.7, linewidth=1.5)
    ax4.plot(df_raw.index, df_raw['Close_EMA13'], label='EMA 13', color='blue', linewidth=2)
    ax4.plot(df_raw.index, df_raw['Close_EMA233'], label='EMA 233', color='violet', linewidth=2)
    ax4.legend(loc='upper left')
    ax4.tick_params(axis='y', labelsize=8)

    # ax4_twin = ax4.twinx()  # Creació d'un segon Eix Y
    # ax4_twin.plot(df_raw.index, df_raw['ATR'], label='ATR', color='green', alpha=0.7, linewidth=1.5)
    # ax4_twin.tick_params(axis='y', labelcolor='green')
    # ax4_twin.legend(['ATR'], loc='lower left')
    # ax4_twin.plot(df_raw.index, df_raw['ATR_Q5'], color='green', linestyle='--', linewidth=0.8)
    # ax4_twin.plot(df_raw.index, df_raw['ATR_Q90'], color='green', linestyle='--', linewidth=0.8)
    # ax4_twin.legend(loc='best')
    # ax4_twin.set_ylabel('RF_Norm (Cost d\'Intervenció)', color='red', fontsize=12)
    # ax4_twin.plot(df_raw.index, df_raw['REPV_R'], color='red', linewidth=1.5, label='REPV_R (Cost)')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return plt.show()


df = df.tail(90)
df_raw = df_raw.tail(72)
df_raw_1 = df_raw_1.tail(72)
grafica(df)
grafica(df_raw)
grafica(df_raw_1)