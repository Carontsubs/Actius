import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# --- Criptos de l'índex ---
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
           "SOL-USD", "DOGE-USD", "DOT-USD", "TRX-USD", "LINK-USD"]

# Actiu a comparar amb l'índex (crypto o índex borsari)
par = "^GSPC"   # 👉 pots canviar-ho per "BNB-USD", "ETH-USD", etc.

# Descarregar dades (últim mes)
data = yf.download(cryptos + [par], period="3mo").dropna()
close_prices = data["Close"]

# Construir Crypto10 Index (equally weighted, normalitzat a 100)
norm = close_prices[cryptos] / close_prices[cryptos].iloc[0] * 100
crypto10 = norm.mean(axis=1)
# Sèrie de l’actiu escollit (forcem que sigui Series)
par_close = close_prices[par].squeeze()

# Normalitzar Crypto10 i l’actiu comparat a 100 al primer dia
crypto10_norm = crypto10 / crypto10.iloc[0] * 100
par_close_norm = par_close / par_close.iloc[0] * 100


# Definir període de la MA
periode_ma = 10

# Calcular MA per Index i per l’actiu comparat
crypto10_ma = crypto10_norm.rolling(window=periode_ma).mean()
par_ma = par_close_norm.rolling(window=periode_ma).mean()

# Ràtio Index / Actiu
ratio = (crypto10_norm / par_close_norm).dropna()

def detectar_inflexio(serie, threshold=0.05):       # threshold=0.01 considera màxim/minim si hi ha 1%
    shift1 = serie.shift(1)
    shift_1 = serie.shift(-1)
    peaks = serie[(serie > shift1 * (1+threshold)) & (serie > shift_1 * (1+threshold))]
    troughs = serie[(serie < shift1 * (1-threshold)) & (serie < shift_1 * (1-threshold))]
    return peaks, troughs

index_peaks, index_troughs = detectar_inflexio(crypto10_norm)
par_peaks, par_troughs = detectar_inflexio(par_close_norm)

# ---- Crear tres subplots ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,14), sharex=True)

# Gràfic 1: Crypto10 Index amb punts d'inflexió
ax1.plot(crypto10_norm, label="Crypto10 Index", color="black")
ax1.plot(index_peaks.index, index_peaks.values, "g^")
ax1.plot(index_troughs.index, index_troughs.values, "rv")
ax1.plot(crypto10_ma, label=f"MA {periode_ma} dies", color="red", linestyle="--")
ax1.legend()

# Gràfic 2: Actiu comparat amb punts d'inflexió
ax2.plot(par_close_norm, label=f"{par} Close", color="blue")
ax2.plot(par_peaks.index, par_peaks.values, "g^")
ax2.plot(par_troughs.index, par_troughs.values, "rv")
ax2.plot(par_ma, label=f"MA {periode_ma} dies", color="red", linestyle="--")
ax2.legend()

# Gràfic 3: Ràtio Index/Actiu
ax3.plot(ratio.index, ratio.values, label=f"Crypto10/{par} Ratio", color="purple")
ax3.axhline(ratio.mean(), color="black", linestyle="--", label="Mitjana")
ax3.legend()

plt.tight_layout()
plt.show()
