import matplotlib.pyplot as plt
import requests
import pandas as pd

url = "https://api.binance.com/api/v3/depth"
params = {"symbol": "POL_USDT", "limit": 10000}
data = requests.get(url, params=params).json()

# Convertim a DataFrame i assegurem columnes correctes
bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'], dtype=float)
asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'], dtype=float)

# Ara sí, podem calcular murs
mur_bids = bids[bids['quantity'] > bids['quantity'].mean() * 100]
mur_asks = asks[asks['quantity'] > asks['quantity'].mean() * 100]

print("\nMurs de compra:")
print(mur_bids)

print("\nMurs de venda:")
print(mur_asks)


plt.figure(figsize=(8,4))
plt.plot(bids['price'], bids['quantity'].cumsum(), label='Bids (compres)', color='green')
plt.plot(asks['price'], asks['quantity'].cumsum(), label='Asks (vendes)', color='red')
plt.axvline(x=(bids['price'].max()+asks['price'].min())/2, color='blue', linestyle='--', label='Mid Price')

plt.xlabel("Preu (USDT)")
plt.ylabel("Volum acumulat")
plt.title("Llibre d'ordres BTC/USDT (Binance)")
plt.legend()
plt.show()
