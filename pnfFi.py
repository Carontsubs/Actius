import yfinance as yf
import pandas as pd
from pypnf import PointFigureChart

# Descarregar dades del ticker
periode = "1y"
interval = "1d"
caixa = 0.02

symbol = 'BTC-USD'

data = yf.Ticker(symbol)
ts = data.history(period=periode, interval=interval)
# ts = data.history(period="1y")

# Preparar les dades
ts.reset_index(level=0, inplace=True)
ts['Date'] = ts['Date'].dt.strftime('%Y-%m-%d')
ts = ts[['Date', 'Open', 'High', 'Low', 'Close']]
ts = ts.to_dict('list')

# # Crear el gràfic P&F
box = round((pd.Series(ts['Close']).mean()*caixa), 5) # Un 2% del preu mitja de tancament de la serie
# box = round((ts['Close'][-1] * 0.02), 3) # Un 2% del preu de tancament ultim
# box = 0.0001
revers = 3
pnf = PointFigureChart(ts=ts, method='cl', reversal=revers, boxsize=box, scaling='abs', title=symbol)
pnf.get_trendlines()

# Mostrar el gràfic
# print(box)
# print(round((pd.Series(ts['Close']).mean() * 0.02), 3))
pnf.show()
