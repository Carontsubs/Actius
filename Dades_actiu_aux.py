import pandas as pd
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

def dades_diaries(df, interval_type='diari'):
    """
    Calcula els indicadors i els llindars dinàmics per a un DataFrame.
    L'argument 'interval_type' s'utilitza per a determinar la finestra de Rolling Quantile.
    Valors possibles: 'daily', '4h', '1h'
    """
    
    # Si les dades no són un MultiIndex (el cas de df_raw), no fem el droplevel
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # df = df[:-1] 
    df = df.copy() # Correcció per evitar el SettingWithCopyWarning
    
    # ----------------------------------------------------------------------
    # --- CÀLCULS DEL SISTEMA 1: VOLATILITAT (ATR / V-ATR) ---
    # ----------------------------------------------------------------------

    df['Prev Close'] = df['Close'].shift(1)
    df['Price_TR'] = abs(df['Close'] - df['Prev Close'])
    df['Prev Volume'] = df['Volume'].shift(1)
    # Línia original: df['Volume_VTR'] = abs(df['Volume'] / df['Prev Volume'])
    # Corregint per evitar divisió per zero si Prev Volume és 0 (poc probable però possible)
    # i per evitar valors extrems si el canvi és molt gran (usar Log o limitar podria ser millor, 
    # però seguim la lògica original, afegint només un petit factor de suavització a l'original)
    df['Volume_VTR'] = abs(df['Volume'] / (df['Prev Volume'].replace(0, 1e-9))) 

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


    # Neteja temporal de NaNs introduïts per les EMAs
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
        # Gestionar la divisió per zero
        denominator_atr_adx = df_adx['ATR_ADX'].replace(0, 1e-9)
        df_adx['+DI'] = 100 * (wilder_smooth(df_adx['+DM'], period) / denominator_atr_adx)
        df_adx['-DI'] = 100 * (wilder_smooth(df_adx['-DM'], period) / denominator_atr_adx)

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
    denominator_atr = df['ATR'].replace(0, 1e-9)
    df['REPV'] = df['SMA_13_Volume'] / denominator_atr
    denominator_vol_ema = df['Volume_ema8'].replace(0, 1e-9)
    df['REPV_a'] = denominator_vol_ema / df['Price_TR_ema8'].replace(0, 1e-9)
    denominator_repv = df['REPV'].replace(0, 1e-9)
    df['REPV_R'] = df['REPV_a'] / denominator_repv

    denominator_repv_r = df['REPV_R'].replace(0, 1e-9)
    df['IPE'] = df['Log_Divergence_Ratio'] / denominator_repv_r

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
    period_atr = 21 
    smooth_k_atr = 3  
    smooth_d_atr = 3

    df['Low_ATR'] = df['ATR'].rolling(window=period_atr).min()
    df['High_ATR'] = df['ATR'].rolling(window=period_atr).max()
    
    # Gestionar la divisió per zero si High_ATR - Low_ATR és 0
    atr_range = df['High_ATR'] - df['Low_ATR']
    df['Fast_%ATR-K'] = np.where(atr_range > 0, 
                                 100 * ((df['ATR'] - df['Low_ATR']) / atr_range), 
                                 50) # Valor per defecte al 50 si no hi ha rang

    df['Slow_%ATR-K'] = df['Fast_%ATR-K'].rolling(window=smooth_k_atr).mean()
    df['Slow_%ATR-D'] = df['Slow_%ATR-K'].rolling(window=smooth_d_atr).mean()


    # Afegir l'ADX calculat al DataFrame principal
    df = df.join(calculate_adx(df), how='left')
    
    # ----------------------------------------------------------------------
    # --- CÀLCUL DE LLINDARS DINÀMICS (ROLLING QUANTILE) ---
    # --- Integració de la lògica de llindars dins la funció ---
    # ----------------------------------------------------------------------

    if interval_type == 'diari':
        WINDOW = 250 # Aproximadament 1 any de trading
    elif interval_type == '4h':
        WINDOW = 72 # Aproximadament 12 dies (6 candeles/dia * 12 dies)
    elif interval_type == '1h':
        WINDOW = 288 # Aproximadament 12 dies (24 candeles/dia * 12 dies)
    else:
        WINDOW = 72 # Default
        
    window = min(WINDOW, len(df))
    
    # Càlcul dels percentils
    df['LDR_Q10'] = df['Log_Divergence_Ratio'].rolling(window=window).quantile(0.10)
    df['LDR_Q90'] = df['Log_Divergence_Ratio'].rolling(window=window).quantile(0.90)
    df['LVR_Q10'] = df['Log_Volatility_Ratio'].rolling(window=window).quantile(0.10)
    df['LVR_Q90'] = df['Log_Volatility_Ratio'].rolling(window=window).quantile(0.90)
    df['REPV_R_Q10'] = df['REPV_R'].rolling(window=window).quantile(0.10)
    df['REPV_R_Q90'] = df['REPV_R'].rolling(window=window).quantile(0.90)
    df['IPE_Q10'] = df['IPE'].rolling(window=window).quantile(0.10)
    df['IPE_Q90'] = df['IPE'].rolling(window=window).quantile(0.90)

    # Neteja final de NaNs introduïts pels Rolling Windows i altres càlculs
    df.dropna(inplace=True) 

    return df


