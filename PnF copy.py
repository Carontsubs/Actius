import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
from telegram import Bot

# Token de acceso del bot (obtenido de BotFather)
TOKEN = '6100990287:AAHXb9-ItOIwEbDVljAbyGWXsRARDrI81P0'

# ID del chat donde quieres enviar los mensajes (puedes obtenerlo hablando con el bot @userinfobot)
CHAT_ID = '5274628'

# Función para enviar una imagen a Telegram
def enviar_imagen_telegram(imagen_path, token, chat_id):
    bot = Bot(token)
    bot.send_photo(chat_id=chat_id, photo=open(imagen_path, 'rb'))

# Definim una funció per calcular el tamany de la caixa segons l'ample de preu
par = 'BTC-USD'

# Definir el periode de temps respecte avui, en dies
dies_enrera = 150

# Calcul de la data d'inci segons el periode de dies estipulat
inici = (datetime.now() - timedelta(days=dies_enrera)).strftime('%Y-%m-%d')

# Definir la fecha final como la fecha actual
final = datetime.now().strftime('%Y-%m-%d')

# Descarrega les dades del BTC
btc_data = yf.download(par, start=inici, end=final)

# Obtenim el darrer preu de tancament del Bitcoin
meitat_longitud = len(btc_data) // 2
preu_referencia = btc_data['Close'].iloc[meitat_longitud]
darrer_preu_tancament = btc_data['Close'].iloc[-1]

def calcular_tamany_caixa(preu):
    if preu < 0.25:
        return 0.0625
    elif 0.25 <= preu < 1.00:
        return 0.125
    elif 1.00 <= preu < 5.00:
        return 0.25
    elif 5.00 <= preu < 20.00:
        return 0.50
    elif 20.00 <= preu < 100:
        return 1
    elif 100 <= preu < 200:
        return 2
    elif 200 <= preu < 500:
        return 4
    elif 500 <= preu < 1000:
        return 5
    elif 1000 <= preu < 25000:
        return 50
    else:
        return 500
    
# Calculem el tamany de caixa i el revers
revers = 3

# box = preu_referencia * 0.1
# box = 1000
box = calcular_tamany_caixa(darrer_preu_tancament) 

imagen_path = 'grafica.png'

# Crea el gràfic punt i figura
hlines=dict(hlines=darrer_preu_tancament, linestyle='--', alpha=1, colors='b', linewidths=1)
mpf.plot(btc_data, type='pnf', hlines=hlines, volume=True, pnf_params=dict(box_size=box, reversal=revers), style='charles', title=f'P&F {par} - {dies_enrera} dies -- Box/Rev ({box}/{revers}) -- Preu({round(darrer_preu_tancament)})', ylabel='Preu (USD)', savefig=imagen_path)

# Enviar la imagen a Telegram
enviar_imagen_telegram(imagen_path, TOKEN, CHAT_ID)
