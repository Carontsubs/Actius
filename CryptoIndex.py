import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Selecció de 10 criptos
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
           "SOL-USD", "DOGE-USD", "DOT-USD", "TRX-USD", "LINK-USD"]

periode = "5d"
interval = "1h"

# Descarregar dades (últim any, diàries)
data = yf.download(cryptos, period=periode, interval=interval)["Close"]

# Eliminar files buides
data = data.dropna()

# Normalitzar (totes comencen a 100)
norm = data / data.iloc[0] * 100

# Índex Crypto10 (mitjana simple)
crypto10 = norm.mean(axis=1)

# --- Gràfics ---
fig, (ax1) = plt.subplots(figsize=(12,14), sharex=True)

# Gràfic 1: Evolució comparada normalitzada
for ticker in norm.columns:   # recorrem totes les criptos
    ax1.plot(norm[ticker], label=ticker, alpha=0.7)

# Afegim l'índex destacat
ax1.plot(crypto10, label="Crypto10 Index", color="black", linewidth=2)

ax1.set_title(f'CryptoIndex10 {periode}/{interval}')

# Llegenda gran a la dreta
ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()

