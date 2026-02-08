import yfinance as yf
import matplotlib.pyplot as plt

# --- Llista de criptos per l'índex ---
cryptos = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
           "SOL-USD", "DOGE-USD", "DOT-USD", "TRX-USD", "LINK-USD"]

par = "TRX-USD"  # cripto de comparació

# --- Descarregar dades ---
data = yf.download(cryptos, period="1y")["Close"].dropna()
data1 = yf.download(par, period="1y")["Close"].dropna()

# --- Construir Crypto10 Index ---
norm = data / data.iloc[0] * 100   # normalitzar totes
crypto10 = norm.mean(axis=1)       # mitjana simple (equally weighted)

# --- Preus de la cripto de comparació ---
par_close = data1[par]
par_norm = par_close / par_close.iloc[0] * 100

# --- Ràtio Crypto10 / comparació ---
ratio = (crypto10 / par_close).dropna()

# --- Tendència Crypto10 ---
preu_final_index = crypto10.iloc[-1]
mitjana_index = crypto10.mean()
percent_diff_index = (preu_final_index - mitjana_index) / mitjana_index * 100

if percent_diff_index > 10:
    tendencia_index = "Alcista"
elif percent_diff_index < -10:
    tendencia_index = "Baixista"
else:
    tendencia_index = "Neutre"

print(f"Tendència Crypto10: {tendencia_index} ({percent_diff_index:.2f}% respecte a la mitjana)")

# --- Tendència de la cripto comparada ---
preu_final_par = par_close.iloc[-1]
mitjana_period_par = par_close.mean()
percent_diff_par = (preu_final_par - mitjana_period_par) / mitjana_period_par * 100

if percent_diff_par > 10:
    tendencia_par = "Alcista"
elif percent_diff_par < -10:
    tendencia_par = "Baixista"
else:
    tendencia_par = "Neutre"

print(f"Tendència {par}: {tendencia_par} ({percent_diff_par:.2f}% respecte a la mitjana)")

# --- Gràfics ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,14), sharex=True)

# Gràfic 1: Evolució comparada normalitzada
ax1.plot(crypto10, label="Crypto10 Index (normalitzat)", color="black", linewidth=2)
ax1.plot(par_norm, label=f"{par} normalitzat", linestyle="--")
ax1.set_title(f"Comparació Crypto10 Index vs {par} (normalitzat)")
ax1.legend()

# Gràfic 2: Ràtio Crypto10 / comparació
ax2.plot(ratio, label=f"Crypto10/{par} Ratio", color="purple")
ax2.axhline(ratio.mean(), color="red", linestyle="--", label="Mitjana")
ax2.set_title(f"Força relativa Crypto10 vs {par}")
ax2.legend()

plt.tight_layout()
plt.show()
