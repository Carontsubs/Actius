import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt  


def veles(par):
    # Definir el periode de temps respecte avui, en dies
    dies_enrera = 90

    # Calcul de la data d'inci segons el periode de dies estipulat
    inici = (datetime.now() - timedelta(days=dies_enrera)).strftime('%Y-%m-%d')

    # Definir la fecha final como la fecha actual
    final = datetime.now().strftime('%Y-%m-%d')

    # Descarrega les dades del BTC
    btc_data = yf.download(par, period='3mo')

    # print(btc_data.head())

    # Si el DataFrame té MultiIndex a les columnes
    btc_data.columns = btc_data.columns.get_level_values(0)  # agafa només la primera capa

    # Sortida: Open, High, Low, Close, Volume, Price (ja plana)
    # print(btc_data.head())


    darrer_preu_tancament = float(btc_data['Close'].iloc[-1])


    # import mplfinance as mpf

    imagen_path = 'grafica.png'

    hlines = dict(
        hlines=[darrer_preu_tancament],   # només els valors
        linestyle='--',
        colors='b',
        linewidths=1,
        alpha=0.5
    )

    mpf.plot(
        btc_data,
        type='candle',
        hlines=hlines,
        volume=True,
        style='charles',
        title=f'{par} ({round(darrer_preu_tancament)})',
        # savefig=imagen_path
    )

    return imagen_path

veles('BTC-USD')