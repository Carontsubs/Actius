
import yfinance as yf
import mplfinance as mpf
import pandas as pd

def pnf(par):
    btc_data = yf.download(par, period='1y', interval='1d').dropna()

    # Si hi ha MultiIndex a les columnes, aplanem
    if isinstance(btc_data.columns, pd.MultiIndex):
        btc_data.columns = btc_data.columns.droplevel(1)

    darrer_preu_tancament = float(btc_data['Close'].iloc[-1])

    def calcular_tamany_caixa(preu):
        if preu < 0.25: return 0.0625
        elif preu < 1.00: return 0.125
        elif preu < 5.00: return 0.25
        elif preu < 20.00: return 0.50
        elif preu < 100: return 1
        elif preu < 200: return 2
        elif preu < 500: return 4
        elif preu < 1000: return 5
        elif preu < 25000: return 50
        else: return 500

    # box = calcular_tamany_caixa(darrer_preu_tancament)
    box = 0.01
    revers = 5

    imagen_path = 'grafica.png'
    hlines = dict(hlines=[darrer_preu_tancament],
                  colors=['b'],
                  linestyle='--',
                  linewidths=1)

    mpf.plot(
        btc_data,
        type='pnf',
        pnf_params=dict(box_size=box, reversal=revers),
        hlines=hlines,
        style='charles',
        title=f'P&F {par} - Box/Rev ({box}/{revers}) - Preu({round(darrer_preu_tancament)})',
        ylabel='Preu (USD)',
    )

    return imagen_path

pnf("POL28321-USD")
