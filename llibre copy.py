import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

class WyckoffAnalyzer:
    def __init__(self, symbol, period="6mo"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.support_levels = []
        self.resistance_levels = []
        self.load_data()
        
    def load_data(self):
        """Carrega les dades històriques"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"No s'han trobat dades per al símbol {self.symbol}")
            print(f"✅ Dades carregades per {self.symbol}: {len(self.data)} registres")
        except Exception as e:
            print(f"❌ Error carregant dades: {e}")

    def calculate_price_patterns(self):
        """Detecta patrons Spring / Upthrust amb suport/resistència del llibre d'ordres"""
        if self.data is None:
            return []

        patterns = []

        # Llibre d'ordres Binance
        url = "https://api.binance.com/api/v3/depth"
        params = {"symbol": "BTCUSDT", "limit": 10000}
        data = requests.get(url, params=params).json()

        bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'], dtype=float)

        # Detectar murs significatius
        mur_bids = bids[bids['quantity'] > bids['quantity'].mean() * 100]  # resistències
        mur_asks = asks[asks['quantity'] > asks['quantity'].mean() * 100]  # suports

        self.resistance_levels = mur_bids['price'].tolist() if not mur_bids.empty else []
        self.support_levels = mur_asks['price'].tolist() if not mur_asks.empty else []

        window = 20
        for i in range(window, len(self.data) - 5):
            current_low = self.data['Low'].iloc[i]
            current_high = self.data['High'].iloc[i]

            # SPRING
            for support in self.support_levels:
                if (current_low < support * 0.995 and
                    self.data['Close'].iloc[i] > current_low * 1.02 and
                    self.data['Volume'].iloc[i] > self.data['Volume'].rolling(10).mean().iloc[i]):
                    if any(self.data['Close'].iloc[j] > support for j in range(i+1, min(i+6, len(self.data)))):
                        patterns.append({
                            'tipus': 'Spring',
                            'data': self.data.index[i],
                            'preu': current_low,
                            'nivell': support,
                            'descripcio': f"Ruptura falsa del suport {support:.2f} amb recuperació"
                        })

            # UPTHRUST
            for resistance in self.resistance_levels:
                if (current_high > resistance * 1.012 and
                    self.data['Close'].iloc[i] < current_high * 0.993 and
                    self.data['Volume'].iloc[i] > self.data['Volume'].rolling(10).mean().iloc[i]):
                    if any(self.data['Close'].iloc[j] < resistance for j in range(i+1, min(i+6, len(self.data)))):
                        patterns.append({
                            'tipus': 'Upthrust',
                            'data': self.data.index[i],
                            'preu': current_high,
                            'nivell': resistance,
                            'descripcio': f"Ruptura falsa de la resistència {resistance:.2f} amb rebuig"
                        })

        return patterns

    def generate_report(self):
        """Mostra l'anàlisi sense plots"""
        print(f"\n📊 INFORME WYCKOFF - {self.symbol}")
        print(f"Data d'anàlisi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Preu actual: ${self.data['Close'].iloc[-1]:.2f}")

        patterns = self.calculate_price_patterns()

        print(f"\n📈 Patrons detectats:")
        if patterns:
            for pat in patterns[-10:]:
                print(f"🎯 {pat['tipus']} - {pat['data'].strftime('%Y-%m-%d')} - ${pat['preu']:.2f} | Nivell: {pat['nivell']:.2f}")
                print(f"   {pat['descripcio']}")
        else:
            print("No s'han detectat patrons significatius.")

        print(f"\n📌 Suports detectats: {self.support_levels}")
        print(f"📌 Resistències detectades: {self.resistance_levels}")

# Exemple d'ús
symbol = "BTC-USD"
analyzer = WyckoffAnalyzer(symbol, "6mo")
analyzer.generate_report()
