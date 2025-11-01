import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import requests

class WyckoffAnalyzer:
    def __init__(self, symbol, period="6mo"):
        """
        Inicialitza l'analitzador Wyckoff
        
        Args:
            symbol (str): Símbol del stock o parell de divises (ex: "AAPL", "EURUSD=X")
            period (str): Període de temps (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Carrega les dades del símbol especificat"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"No s'han trobat dades per al símbol {self.symbol}")
            print(f"✅ Dades carregades per {self.symbol}: {len(self.data)} registres")
        except Exception as e:
            print(f"❌ Error carregant dades: {e}")
            
    def calculate_volume_metrics(self):
        """Calcula mètriques de volum"""
        if self.data is None:
            return None
            
        volume_data = {
            'volum_actual': self.data['Volume'].iloc[-1],
            'volum_mitjana_20': self.data['Volume'].rolling(20).mean().iloc[-1],
            'volum_mitjana_50': self.data['Volume'].rolling(50).mean().iloc[-1],
            'volum_relatiu': self.data['Volume'].iloc[-1] / self.data['Volume'].rolling(20).mean().iloc[-1],
            'volum_maxim_periode': self.data['Volume'].max(),
            'volum_minim_periode': self.data['Volume'].min()
        }
        
        return volume_data
    
    def calculate_price_patterns(self):
        """Detecta patrons de preu fent servir suports/resistències del llibre d’ordres"""
        if self.data is None:
            return None

        patterns = []

        # 1. Llibre d'ordres Binance
        url = "https://api.binance.com/api/v3/depth"
        params = {"symbol": "BTCUSDT", "limit": 1000}
        data = requests.get(url, params=params).json()

        bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'], dtype=float)

        # 2. Detectar murs significatius
        mur_bids = bids[bids['quantity'] > bids['quantity'].mean() * 50]  # resistències
        mur_asks = asks[asks['quantity'] > asks['quantity'].mean() * 50]  # suports

        resistance_levels = [float(x) for x in mur_bids['price']] if not mur_bids.empty else []
        support_levels = [float(x) for x in mur_asks['price']] if not mur_asks.empty else []

        window = 20

        # 3. Analitzar si hi ha patrons Spring / Upthrust contra aquests nivells
        for i in range(window, len(self.data) - 5):
            current_low = self.data['Low'].iloc[i]
            current_high = self.data['High'].iloc[i]

            # SPRING → toca un suport i recupera
            for support in support_levels:
                try:
                    if (current_low < support * 0.995 and
                        self.data['Close'].iloc[i] > current_low * 1.01 and
                        self.data['Volume'].iloc[i] > self.data['Volume'].rolling(10).mean().iloc[i]):

                        if any(self.data['Close'].iloc[j] > support for j in range(i+1, min(i+6, len(self.data)))):
                            patterns.append({
                                'tipus': 'Spring',
                                'data': self.data.index[i],
                                'preu': current_low,
                                'nivell': support,
                                'descripcio': f"Ruptura falsa del suport {support:.2f} amb recuperació"
                            })
                except:
                    continue

            # UPTHRUST → toca una resistència i rebutja
            for resistance in resistance_levels:
                try:
                    if (current_high > resistance * 1.005 and
                        self.data['Close'].iloc[i] < current_high * 0.995 and
                        self.data['Volume'].iloc[i] > self.data['Volume'].rolling(10).mean().iloc[i]):

                        if any(self.data['Close'].iloc[j] < resistance for j in range(i+1, min(i+6, len(self.data)))):
                            patterns.append({
                                'tipus': 'Upthrust',
                                'data': self.data.index[i],
                                'preu': current_high,
                                'nivell': resistance,
                                'descripcio': f"Ruptura falsa de la resistència {resistance:.2f} amb rebuig"
                            })
                except:
                    continue

        return patterns, support_levels, resistance_levels
    
    def calculate_trading_range(self):
        """Calcula el rang de negociació actual"""
        if self.data is None:
            return None
            
        # Rang dels últims 50 períodes
        recent_data = self.data.tail(int(50))
        
        trading_range = {
            'maxim_rang': recent_data['High'].max(),
            'minim_rang': recent_data['Low'].min(),
            'amplada_rang': recent_data['High'].max() - recent_data['Low'].min(),
            'preu_actual': self.data['Close'].iloc[-1],
            'posicio_rang': ((self.data['Close'].iloc[-1] - recent_data['Low'].min()) / 
                            (recent_data['High'].max() - recent_data['Low'].min())) * 100,
            'volatilitat': recent_data['High'].sub(recent_data['Low']).mean()
        }
        
        return trading_range
    
    def identify_market_phase(self):
        """Identifica les fases de mercat segons Wyckoff"""
        if self.data is None:
            return None
            
        # Calcular indicadors tècnics
        self.data['SMA_20'] = self.data['Close'].rolling(20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(50).mean()
        self.data['Volume_SMA'] = self.data['Volume'].rolling(20).mean()
        
        recent_data = self.data.tail(30)
        current_price = self.data['Close'].iloc[-1]
        sma_20 = self.data['SMA_20'].iloc[-1]
        sma_50 = self.data['SMA_50'].iloc[-1]
        
        # Anàlisi de tendència i volum
        price_trend = "Alcista" if current_price > sma_20 > sma_50 else "Baixista" if current_price < sma_20 < sma_50 else "Lateral"
        
        volume_trend = "Creixent" if recent_data['Volume'].iloc[-10:].mean() > recent_data['Volume'].iloc[-20:-10].mean() else "Decreixent"
        
        # Determinació de la fase
        if price_trend == "Lateral" and volume_trend == "Creixent":
            if current_price < (recent_data['High'].max() + recent_data['Low'].min()) / 2:
                phase = "Acumulació"
                description = "Possible fase d'acumulació - preus laterals amb volum creixent a la part baixa del rang"
            else:
                phase = "Distribució"
                description = "Possible fase de distribució - preus laterals amb volum creixent a la part alta del rang"
        elif price_trend == "Alcista" and volume_trend == "Creixent":
            phase = "Mark-up"
            description = "Fase de Mark-up - tendència alcista amb volum de confirmació"
        elif price_trend == "Baixista" and volume_trend == "Creixent":
            phase = "Mark-down"
            description = "Fase de Mark-down - tendència baixista amb volum de confirmació"
        else:
            phase = "Indeterminada"
            description = "Fase no clara - cal més observació"
            
        return {
            'fase': phase,
            'descripcio': description,
            'tendencia_preu': price_trend,
            'tendencia_volum': volume_trend
        }
    
    def analyze_supply_demand(self):
        """Analitza l'oferta i demanda relatives"""
        if self.data is None:
            return None
            
        # Calcular indicadors d'oferta i demanda
        recent_data = self.data.tail(20)
        
        # Pressió compradora vs venedora basada en tancaments
        up_days = (recent_data['Close'] > recent_data['Open']).sum()
        down_days = (recent_data['Close'] < recent_data['Open']).sum()
        
        # Volum en dies alcistes vs baixistes
        up_volume = recent_data[recent_data['Close'] > recent_data['Open']]['Volume'].sum()
        down_volume = recent_data[recent_data['Close'] < recent_data['Open']]['Volume'].sum()
        
        # Anàlisi de spreads (diferència high-low)
        avg_spread = recent_data['High'].sub(recent_data['Low']).mean()
        recent_spread = self.data['High'].iloc[-1] - self.data['Low'].iloc[-1]
        
        # Effort vs Result
        price_change = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[-20]) / self.data['Close'].iloc[-20] * 100
        volume_change = (recent_data['Volume'].mean() - self.data['Volume'].tail(40).head(20).mean()) / self.data['Volume'].tail(40).head(20).mean() * 100
        
        supply_demand = {
            'dies_alcistes': up_days,
            'dies_baixistes': down_days,
            'ratio_dies': up_days / max(down_days, 1),
            'volum_alcista': up_volume,
            'volum_baixista': down_volume,
            'ratio_volum': up_volume / max(down_volume, 1),
            'spread_mitjà': avg_spread,
            'spread_actual': recent_spread,
            'canvi_preu_20d': price_change,
            'canvi_volum_20d': volume_change,
            'effort_vs_result': 'Eficient' if abs(price_change) > 2 and volume_change > 10 else 
                               'No confirmat' if abs(price_change) < 1 and volume_change > 10 else 'Normal'
        }
        
        # Interpretació
        if supply_demand['ratio_volum'] > 1.5:
            supply_demand['interpretacio'] = "Demanda dominant"
        elif supply_demand['ratio_volum'] < 0.67:
            supply_demand['interpretacio'] = "Oferta dominant"
        else:
            supply_demand['interpretacio'] = "Equilibri relatiu"
            
        return supply_demand
    
    def generate_report(self):
        """Genera un informe complet de l'anàlisi"""
        print(f"\n{'='*60}")
        print(f"📊 INFORME D'ANÀLISI WYCKOFF - {self.symbol}")
        print(f"{'='*60}")
        print(f"Data d'anàlisi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Període analitzat: {self.period}")
        print(f"Preu actual: ${self.data['Close'].iloc[-1]:.2f}")
        
        # Mètriques de volum
        print(f"\n🔊 ANÀLISI DE VOLUM")
        print("-" * 30)
        volume_metrics = self.calculate_volume_metrics()
        if volume_metrics:
            print(f"Volum actual: {volume_metrics['volum_actual']:,.0f}")
            print(f"Mitjana 20 períodes: {volume_metrics['volum_mitjana_20']:,.0f}")
            print(f"Volum relatiu: {volume_metrics['volum_relatiu']:.2f}x")
            if volume_metrics['volum_relatiu'] > 1.5:
                print("⚡ Volum elevat - Possible activitat institucional")
            elif volume_metrics['volum_relatiu'] < 0.5:
                print("💤 Volum baix - Falta d'interès")
        
        # Patrons de preu
        print(f"\n📈 PATRONS DE PREU DETECTATS")
        print("-" * 35)
        patterns = self.calculate_price_patterns()
        if patterns:
            for pattern in patterns[-5:]:  # Últims 5 patrons
                print(f"🎯 {pattern['tipus']} - {pattern['data'].strftime('%Y-%m-%d')} - ${pattern['preu']:.2f}")
                print(f"   {pattern['descripcio']}")
        else:
            print("No s'han detectat patrons significatius recentment")
        
        # Rang de negociació
        print(f"\n📏 RANG DE NEGOCIACIÓ")
        print("-" * 25)
        trading_range = self.calculate_trading_range()
        if trading_range:
            print(f"Rang: ${trading_range['minim_rang']:.2f} - ${trading_range['maxim_rang']:.2f}")
            print(f"Amplada: ${trading_range['amplada_rang']:.2f}")
            print(f"Posició actual: {trading_range['posicio_rang']:.1f}% del rang")
            print(f"Volatilitat mitjana: ${trading_range['volatilitat']:.2f}")
        
        # Fase de mercat
        print(f"\n🌊 FASE DE MERCAT (WYCKOFF)")
        print("-" * 32)
        market_phase = self.identify_market_phase()
        if market_phase:
            print(f"Fase identificada: {market_phase['fase']}")
            print(f"Descripció: {market_phase['descripcio']}")
            print(f"Tendència preu: {market_phase['tendencia_preu']}")
            print(f"Tendència volum: {market_phase['tendencia_volum']}")
        
        # Oferta i demanda
        print(f"\n⚖️ ANÀLISI OFERTA/DEMANDA")
        print("-" * 30)
        supply_demand = self.analyze_supply_demand()
        if supply_demand:
            print(f"Dies alcistes vs baixistes: {supply_demand['dies_alcistes']} vs {supply_demand['dies_baixistes']}")
            print(f"Ratio volum alcista/baixista: {supply_demand['ratio_volum']:.2f}")
            print(f"Interpretació: {supply_demand['interpretacio']}")
            print(f"Effort vs Result: {supply_demand['effort_vs_result']}")
            print(f"Canvi preu (20d): {supply_demand['canvi_preu_20d']:+.2f}%")
            print(f"Canvi volum (20d): {supply_demand['canvi_volum_20d']:+.2f}%")
        
        print(f"\n{'='*60}")
        print("✅ Anàlisi completada!")
        
    def plot_analysis(self):
        """Crea gràfics de l'anàlisi"""
        if self.data is None:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Anàlisi Wyckoff - {self.symbol}', fontsize=16, fontweight='bold')
        
        # Gràfic de preus amb patrons
        ax1.plot(self.data.index, self.data['Close'], label='Preu de tancament', linewidth=1.5)
        ax1.plot(self.data.index, self.data['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA 50', alpha=0.7)
        
        patterns = self.calculate_price_patterns()
        if patterns:
            for pattern in patterns[-10:]:
                color = 'green' if pattern['tipus'] == 'Spring' else 'red'
                ax1.scatter(pattern['data'], pattern['preu'], 
                           color=color, s=100, marker='^' if pattern['tipus'] == 'Spring' else 'v',
                           label=pattern['tipus'] if pattern == patterns[-1] else "")
        
        ax1.set_title('Preus i Patrons')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gràfic de volum
        ax2.bar(self.data.index, self.data['Volume'], alpha=0.6, 
                color=['green' if c > o else 'red' for c, o in zip(self.data['Close'], self.data['Open'])])
        ax2.plot(self.data.index, self.data['Volume_SMA'], color='blue', label='Mitjana volum')
        ax2.set_title('Volum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gràfic de rang de negociació
        recent_data = self.data.tail(50)
        ax3.fill_between(recent_data.index, recent_data['Low'], recent_data['High'], alpha=0.3)
        ax3.plot(recent_data.index, recent_data['Close'], color='black', linewidth=2)
        ax3.axhline(y=recent_data['High'].max(), color='red', linestyle='--', alpha=0.7, label='Resistència')
        ax3.axhline(y=recent_data['Low'].min(), color='green', linestyle='--', alpha=0.7, label='Suport')
        ax3.set_title('Rang de Negociació (50 períodes)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gràfic d'oferta/demanda
        supply_demand = self.analyze_supply_demand()
        if supply_demand:
            categories = ['Dies\nAlcistes', 'Dies\nBaixistes', 'Volum\nAlcista', 'Volum\nBaixista']
            values = [supply_demand['dies_alcistes'], supply_demand['dies_baixistes'],
                     supply_demand['volum_alcista']/1000000, supply_demand['volum_baixista']/1000000]
            colors = ['green', 'red', 'lightgreen', 'lightcoral']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_title('Oferta vs Demanda')
            ax4.grid(True, alpha=0.3)
            
            # Afegir valors a les barres
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}M' if 'Volum' in categories[bars.index(bar)] else f'{int(value)}',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def main():
    """Funció principal per executar l'analitzador"""
    print("🎯 ANALITZADOR WYCKOFF")
    print("=" * 50)
    
    # Exemples de símbols
    print("\nExemples de símbols:")
    print("- Stocks: AAPL, MSFT, GOOGL, TSLA, NVDA")
    print("- Criptodivises: BTC-USD, ETH-USD")
    print("- Forex: EURUSD=X, GBPUSD=X, USDJPY=X")
    print("- Índexs: ^GSPC (S&P500), ^DJI (Dow Jones)")
    
    symbol = input("\nIntrodueix el símbol a analitzar: ").upper()
    
    print("\nPeríodes disponibles:")
    print("1d, 5d, 1mo, 3mo, 6mo (defecte), 1y, 2y, 5y")
    period = input("Període (premeu Enter per 6mo): ").strip()
    if not period:
        period = "6mo"
    
    try:
        analyzer = WyckoffAnalyzer(symbol, period)
        analyzer.generate_report()
        
        show_chart = input("\nVols veure els gràfics? (s/n): ").lower().startswith('s')
        if show_chart:
            analyzer.plot_analysis()
            
    except Exception as e:
        print(f"❌ Error durant l'anàlisi: {e}")

# Executar el programa
if __name__ == "__main__":
    main() 