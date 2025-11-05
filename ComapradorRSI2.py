import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# --- Criptos de l'índex ---
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
           "SOL-USD", "DOGE-USD", "DOT-USD", "XRP-USD", "LINK-USD"]

# Cripto a comparar amb l'índex
par = "DOT-USD"   # pots canviar-ho per "BNB-USD", "SOL-USD", etc.

# Descarregar dades (últim mes)
data = yf.download(cryptos, period="1mo").dropna()
close_prices = data["Close"]

# Construir Crypto10 Index (equally weighted)
norm = close_prices / close_prices.iloc[0] * 100
crypto10 = norm.mean(axis=1)

# Sèrie de la cripto escollida
par_close = close_prices[par]

# Definir període de la MA
periode_ma = 7

# Calcular MA per Index i per la cripto
crypto10_ma = crypto10.rolling(window=periode_ma).mean()
par_ma = par_close.rolling(window=periode_ma).mean()

# Ràtio Index / Cripto
ratio = (crypto10 / par_close).dropna()

# ---- Detectar punts d'inflexió (sense SciPy) ----
def detectar_inflexio(serie):
    shift1 = serie.shift(1)
    shift_1 = serie.shift(-1)
    peaks = serie[(serie > shift1) & (serie > shift_1)]
    troughs = serie[(serie < shift1) & (serie < shift_1)]
    return peaks, troughs

index_peaks, index_troughs = detectar_inflexio(crypto10)
par_peaks, par_troughs = detectar_inflexio(par_close)

# ---- Crear tres subplots ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,14), sharex=True)

# Gràfic 1: Crypto10 Index amb punts d'inflexió
ax1.plot(crypto10, label="Crypto10 Index", color="black")
ax1.plot(index_peaks.index, index_peaks.values, "g^", label="Màxim local")
ax1.plot(index_troughs.index, index_troughs.values, "rv", label="Mínim local")
ax1.set_title("Crypto10 Index amb punts d'inflexió")
ax1.legend()

# Gràfic 2: Cripto comparada amb punts d'inflexió
ax2.plot(par_close, label=f"{par} Close", color="blue")
ax2.plot(par_peaks.index, par_peaks.values, "g^", label="Màxim local")
ax2.plot(par_troughs.index, par_troughs.values, "rv", label="Mínim local")
ax2.set_title(f"{par} amb punts d'inflexió")

# Gràfic 3: Ràtio Index/Cripto
ax3.plot(ratio, label=f"Crypto10/{par} Ratio", color="purple")
ax3.axhline(ratio.mean(), color="red", linestyle="--", label="Mitjana")
ax3.set_title(f"Força relativa Crypto10 vs {par}")
ax3.legend()

plt.tight_layout()
plt.show()
