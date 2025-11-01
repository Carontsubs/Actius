import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta


# Definir el periode de temps respecte avui, en dies
dies_enrera = 30

# Calcul de la data d'inci segons el periode de dies estipulat
inici = (datetime.now() - timedelta(days=dies_enrera)).strftime('%Y-%m-%d')

# Definir la fecha final como la fecha actual
final = datetime.now().strftime('%Y-%m-%d')

# Descarregar dades històriques
data = yf.download("BTC-USD", start=inici, end=final)
# Eliminar multiindex de columnes
data.columns = data.columns.droplevel(1)
data.index.name = "Date"

# Calcula el RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcula l'ADX
def calculate_adx(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    plus_dm = high.diff().where(high.diff() > low.diff(), 0)
    minus_dm = low.diff().where(low.diff() > high.diff(), 0)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx

# Calcula el MACD
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Calcula l'OBV
def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

# Afegir indicadors tècnics al DataFrame
data['RSI'] = calculate_rsi(data)
data['ADX'] = calculate_adx(data)
data['MACD'], data['MACD_Signal'] = calculate_macd(data)
data['OBV'] = calculate_obv(data)

# Crear subplots per visualitzar els indicadors
apds = [
    mpf.make_addplot(data['RSI'], panel=1, color='b', secondary_y=False, ylabel='RSI'),
    mpf.make_addplot(data['ADX'], panel=2, color='g', secondary_y=False, ylabel='ADX'),
    mpf.make_addplot(data['MACD'], panel=3, color='r', secondary_y=False, ylabel='MACD'),
    mpf.make_addplot(data['MACD_Signal'], panel=3, color='b', secondary_y=False),
    mpf.make_addplot(data['OBV'], panel=4, color='purple', secondary_y=False, ylabel='OBV')
]


# # Crear el gràfic amb veles i OBV
# apds = [mpf.make_addplot(data['OBV'], color='orange', secondary_y=True, ylabel='OBV')]

# Configurar l'estil de la trama
style = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mpf.make_marketcolors(up='red', down='red', edge='red', wick='red', volume='blue'))

# Crear el gràfic amb veles i OBV en el mateix plot
mpf.plot(data, type='line', linecolor='black', style=style, addplot=apds, volume=True, title=f'BTC-USD {dies_enrera} dies', ylabel='Preu')

# # Crear el gràfic
mpf.plot(data, type='candle', style='charles', addplot=apds, volume=True, title='BTC-USD amb Indicadors', ylabel='Preu')

