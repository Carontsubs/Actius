import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#Definir el parell
parell = 'DOGE-USD'

# Definir el periode de temps respecte avui, en dies
dies_enrera = 90

# Calcul de la data d'inci segons el periode de dies estipulat
inici = (datetime.now() - timedelta(days=dies_enrera)).strftime('%Y-%m-%d')

# Definir la fecha final como la fecha actual
final = datetime.now().strftime('%Y-%m-%d')

# Obtener datos del Bitcoin
crypto_data = yf.download(parell, start=inici, end=final)

crypto_data.columns = crypto_data.columns.droplevel(1)
par_growth = crypto_data['Close'].pct_change() * 100  # Crecimiento porcentual de ETH-USD

# Definir la función para calcular el MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['ShortEMA'] = df['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    df['LongEMA'] = df['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    df['MACD'] = df['ShortEMA'] - df['LongEMA']
    df['SignalLine'] = df['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return df.drop(['ShortEMA', 'LongEMA'], axis=1)

# Definir la función para calcular el MFI
def calculate_mfi(df, n=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    df['MoneyFlowPositive'] = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    df['MoneyFlowNegative'] = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    money_flow_ratio = df['MoneyFlowPositive'].rolling(window=n).sum() / df['MoneyFlowNegative'].rolling(window=n).sum()
    df['MFI'] = 100 - (100 / (1 + money_flow_ratio))
    return df.drop(['MoneyFlowPositive', 'MoneyFlowNegative'], axis=1)


# Definir la función para calcular el ADX, ADI+ y ADI-
def calculate_adx(df, n=7):    
    # 1. Calcular els moviments (UP i DOWN)
    up = df['High'] - df['High'].shift()
    down = df['Low'].shift() - df['Low']
    df['TR'] = np.max([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=0)
    df['DMplus'] = np.where((up > down) & (up > 0), up, 0)
    df['DMminus'] = np.where((down > up) & (down > 0), down, 0)    
    df['TRn'] = df['TR'].rolling(window=n).mean()
    df['DMplusn'] = df['DMplus'].rolling(window=n).mean()
    df['DMminusn'] = df['DMminus'].rolling(window=n).mean()
    df['DIplus'] = 100 * df['DMplusn'] / df['TRn']
    df['DIminus'] = 100 * df['DMminusn'] / df['TRn']
    df['DX'] = 100 * np.abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
    df['ADX'] = df['DX'].rolling(window=n).mean()
    return df

# Calcular la media móvil llarga
MA = int(dies_enrera / 5)
if dies_enrera > 200:
    MA = 200
if dies_enrera > 50:
    btc_ma_50 = crypto_data['Close'].rolling(window=50).mean()
    

btc_ma_15 = crypto_data['Close'].rolling(window=MA).mean()

# Calcular el MFI
crypto_data = calculate_adx(crypto_data)
crypto_data = calculate_mfi(crypto_data)

# Calcular el MACD
crypto_data = calculate_macd(crypto_data)

high = crypto_data['High'].squeeze()
low = crypto_data['Low'].squeeze()

fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
# Graficar el precio y el spread
axes[0].plot(crypto_data.index, crypto_data['Close'], color='black', label=f'Preu {parell} {dies_enrera} dies')
# axes[0].plot(btc_futures.index, btc_futures['Close'], label='Futurs de Bitcoin (BTC=F)', color='blue')
axes[0].fill_between(crypto_data.index, high, low, color='cyan', alpha=0.2, label='Spread')
axes[0].plot(btc_ma_15.index, btc_ma_15, label=f'MA {MA}', color='red')
axes[0].plot(btc_ma_50.index, btc_ma_50, label=f'MA 50', color='orange')
axes[0].set_ylabel('Precio')
axes[0].grid(True)
axes[0].legend()

# Graficar el volumen
axes[1].plot    (crypto_data.index, crypto_data['Volume'].squeeze().astype(float).values, color='blue', alpha=0.5)
axes[1].set_ylabel('Volumen')
axes[1].grid(True)

# Graficar el MACD
axes[2].plot(crypto_data.index, crypto_data['MACD'], color='orange', label='MACD(12,26)')
axes[2].plot(crypto_data.index, crypto_data['SignalLine'], color='blue', label='Signal Line(9)')
axes[2].fill_between(crypto_data.index, crypto_data['MACD']-crypto_data['SignalLine'], color='gray', alpha=0.3, where=(crypto_data['MACD']-crypto_data['SignalLine']>0), interpolate=True)
axes[2].fill_between(crypto_data.index, crypto_data['MACD']-crypto_data['SignalLine'], color='red', alpha=0.3, where=(crypto_data['MACD']-crypto_data['SignalLine']<=0), interpolate=True)
axes[2].set_ylabel('MACD')
axes[2].grid(True)
axes[2].legend()


# # Graficar la diferencia de crecimiento porcentual
# axes[3].plot(diferencia_crecimiento.index, diferencia_crecimiento, color='black', label='%Abs')
# axes[3].plot(diferencia_crecimiento.index, diferencia_crecimiento_2, color='red', linestyle='--', label='%Rel')
# # axes[3].set_xlabel('Fecha')
# axes[3].set_ylabel('Diferencia del %')
# axes[3].set_title(f'Variacio diaria %BTC-USD - %{parell}')
# axes[3].grid(True)
# axes[3].legend()

# # Graficar el ADI+ y ADI-
# Graficar el ADX
axes[3].plot(crypto_data.index, crypto_data['ADX'], color='red', label='ADX')
axes[3].plot(crypto_data.index, crypto_data['DIplus'], color='green', label='ADI+')
axes[3].plot(crypto_data.index, crypto_data['DIminus'], color='blue', label='ADI-')
axes[3].set_ylabel('ADX i DI')
axes[3].grid(True)
axes[3].legend()
# axes[3].set_ylabel('DI')

# Graficar el MFI
axes[4].plot(crypto_data.index, crypto_data['MFI'], color='green')
axes[4].axhline(y=80, color='r', linestyle='--')  # Línea de sobrecompra
axes[4].axhline(y=20, color='b', linestyle='--')  # Línea de sobreventa
axes[4].set_ylabel('MFI')
axes[4].grid(True)



# Ajustar el diseño de los subplots
plt.tight_layout()

# Mostrar el gráfico
plt.show()
