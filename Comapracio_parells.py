import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


#Definir el parell
par1 = 'BTC-USD'
par2 = 'BNB-USD'

MA = 10

# Definir el periode de temps respecte avui, en dies
dies_enrera = 100

# Calcul de la data d'inci segons el periode de dies estipulat
inici = (datetime.now() - timedelta(days=dies_enrera)).strftime('%Y-%m-%d')

# Definir la fecha final como la fecha actual
final = datetime.now().strftime('%Y-%m-%d')


# Obtener datos de BTC-USD e ETH-USD
btc_data = yf.download(par1, start=inici, end=final)
par_data = yf.download(par2, start=inici, end=final)

# Calcular la media móvil de 15 días
btc_ma_15 = btc_data['Close'].rolling(window=MA).mean()
par_ma_15 = par_data['Close'].rolling(window=MA).mean()

# Calcular la diferencia de crecimiento porcentual entre BTC-USD y ETH-USD en dos días consecutivos
btc_growth = btc_data['Close'].pct_change() * 100  # Crecimiento porcentual de BTC-USD
par_growth = par_data['Close'].pct_change() * 100  # Crecimiento porcentual de ETH-USD
diferencia_crecimiento = abs(btc_growth) - abs(par_growth)   # Diferencia de crecimiento porcentual entre los dos pares
diferencia_crecimiento_2 = btc_growth - par_growth   # Diferencia de crecimiento porcentual entre los dos pares


# Graficar los precios de cierre de BTC-USD y ETH-USD en gráficos separados
fig, axs =plt.subplots(3, 1, figsize=(12, 10))

# Gráfico para los precios de cierre de BTC-USD
axs[0].plot(btc_data.index, btc_data['Close'], label=f'{par1}', color='blue')
axs[0].fill_between(btc_data.index, btc_data['High'], btc_data['Low'], color='cyan', alpha=0.2, label='Spread')
axs[0].plot(btc_ma_15.index, btc_ma_15, label=f'MA {MA}', color='red')
axs[0].set_ylabel(f'Precio de Cierre {par1}')
axs[0].grid(True)
axs[0].legend()

# Gráfico para los precios de cierre de ETH-USD
axs[1].plot(par_data.index, par_data['Close'], label=par2, color='orange')
axs[1].fill_between(par_data.index, par_data['High'], par_data['Low'], color='gold', alpha=0.5, label='Spread')
axs[1].plot(par_data.index, par_ma_15, label=f'MA {MA}', color='red')
axs[1].set_ylabel(f'Precio de Cierre {par2}')
axs[1].grid(True)
axs[1].legend()

# Graficar la diferencia de crecimiento porcentual
axs[2].plot(diferencia_crecimiento.index, diferencia_crecimiento, color='black', label='%Abs')
axs[2].plot(diferencia_crecimiento.index, diferencia_crecimiento_2, color='red', linestyle='--', label='%')
# axs[2].set_xlabel('Fecha')
axs[2].set_ylabel('Diferencia del %')
axs[2].set_title(f'%{par1} - %{par2}')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()

plt.show()
