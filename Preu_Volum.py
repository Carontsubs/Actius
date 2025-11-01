import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Definir el ticker
ticker = 'BTC-USD'

# Función para obtener datos y calcular las MA de precios y volumen
def obtenir_dades_amb_MA(ticker, ma_period):
    # Obtener los datos históricos del último periodo
    data = yf.Ticker(ticker).history(period=f'{ma_period*5}d')
    # Calcular la MA del precio de cierre y del volumen
    data['MA_Close'] = data['Close'].rolling(window=ma_period).mean()
    data['MA_Volume'] = data['Volume'].rolling(window=ma_period).mean()

    # Crear la figura y el eje
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Graficar los precios de cierre y su MA
    color = 'tab:blue'
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio de Cierre', color=color)
    ax1.plot(data.index, data['Close'], label='Close', color=color)
    ax1.plot(data.index, data['MA_Close'], label=f'MA {ma_period} Close', color='tab:red')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Crear un segundo eje para el volumen
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('Volumen', color=color)
    ax2.plot(data.index, data['Volume'], label='Volume', color='blue')
    ax2.fill_between(data.index, data['Volume'], where=data['Volume']>=0, color='skyblue', alpha=0.2)
    ax2.plot(data.index, data['MA_Volume'], label=f'MA {ma_period} Volume', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Título y leyenda
    plt.title(f'{ticker} - Precio de Cierre y Volumen {ma_period*5}dies')
    fig.tight_layout()  
    plt.show()

    # Obtener los valores finales para la salida en texto
    ma_close = data['MA_Close'][-1]
    ma_volume = data['MA_Volume'][-1]
    
    return f"{ticker}: MA Precios: ${round(ma_close, 2)}   MA Volumen: {round(ma_volume)}"

# Ejecutar la función y obtener los datos
ma_period = 30  # Definir el periodo de la MA
result = obtenir_dades_amb_MA(ticker, ma_period)

# Imprimir los resultados
print(result)
