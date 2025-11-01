import yfinance as yf
import matplotlib.pyplot as plt

par = "BNB-USD"

# Descarregar dades de BTC i ETH
data = yf.download(["BTC-USD", par], period="1y").dropna()

# Crear sèries de tancament
btc_close = data["Close"]["BTC-USD"]
par_close = data["Close"][par]

# Normalitzar per comparar (punt de partida = 100)
btc_norm = btc_close / btc_close.iloc[0] * 100
par_norm = par_close / par_close.iloc[0] * 100

# Calcular ràtio de força relativa BTC/ETH
ratio = (btc_close / par_close).dropna()

# # Debuggin
# print(type(btc_close))
# print(type(par_close))
# print(type(ratio))
# print(ratio.head())

# ---- Determinar tendència del BTC ----
preu_final = btc_close.iloc[-1]
mitjana_period = btc_close.mean()

percent_diff = (preu_final - mitjana_period) / mitjana_period * 100

if percent_diff > 10:
    tendencia = "Alcista"
elif percent_diff < -10:
    tendencia = "Baixista"
else:
    tendencia = "Neutre"

print(f"Tendència BTC: {tendencia} ({percent_diff:.2f}% respecte a la mitjana)")

# ---- Determinar tendència del BTC ----
preu_final_par = par_close.iloc[-1]
mitjana_period_par = par_close.mean()

percent_diff_par = (preu_final_par - mitjana_period_par) / mitjana_period_par * 100

if percent_diff_par > 10:
    tendencia = "Alcista"
elif percent_diff_par < -10:
    tendencia = "Baixista"
else:
    tendencia = "Neutre"

print(f"Tendència {par}: {tendencia} ({percent_diff_par:.2f}% respecte a la mitjana)")

# ---- CREAR FINESTRA AMB TRES SUBPLOTS ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,14), sharex=True)

# # Gràfic 1: Preu BTC (Close)
# ax1.plot(btc_close, label="BTC Close", color="blue")
# ax1.set_title("Preu BTC (Close)")
# ax1.legend()

# Gràfic 2: Evolució comparada normalitzada
ax1.plot(btc_norm, label="BTC normalitzat")
ax1.plot(par_norm, label=f"{par} normalitzat")
ax1.set_title(f"Comparació BTC vs {par} (normalitzat)")
ax1.legend()

# Gràfic 3: Ràtio BTC/ETH
ax2.plot(ratio, label=f"BTC/{par} Ratio", color="purple")
ax2.axhline(ratio.mean(), color="red", linestyle="--", label="Mitjana")
ax2.set_title(f"Força relativa BTC vs {par}")
ax2.legend()

# Ajustar disseny i mostrar
plt.tight_layout()
plt.show()
