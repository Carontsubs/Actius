import yfinance as yf
from google import genai
from google.genai import types
import os # Importem el mòdul os per accedir a les variables d'entorn
from dotenv import load_dotenv # Importem la funció per carregar .env
import requests
import Dades_actiu_aux as aux

# Carrega les variables d'entorn del fitxer .env
load_dotenv() 

# Obtenim la clau d'API. Si no la troba, generarà un error.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOKEN_TELEGRAM = os.getenv("TOKEN_TELEGRAM")
CHAT_ID =  os.getenv("CHAT_ID")

def envia_missatge(text):
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": text}
    resposta = requests.get(url, params=params)
    # print(f"Estat de l'enviament a Telegram: {resposta.json()}")

# 1. Extreure les dades de Yahoo Finance
ticker = "BTC-USD"

# Descarregar les dades històriques
df = yf.download(ticker, period="2y", interval='1d')
df_raw_1 = yf.download(ticker, period="3mo", interval='1h')
df_raw = yf.download(ticker, period="3mo", interval='4h')

df = aux.dades_diaries(df,interval_type='diari')
df_raw = aux.dades_diaries(df_raw,interval_type='4h')
df_raw_1 = aux.dades_diaries(df_raw_1, interval_type='1h')

df = df.tail(90)
df_raw = df_raw.tail(72)
df_raw_1 = df_raw_1.tail(72)

# Inicialitza el client passant la clau directament.
client = genai.Client(api_key=GEMINI_API_KEY)
print(f"Generant informe...")  # --- 3. CONSTRUCCIÓ DEL PROMPT FINAL ---

prompt = f"""
  [ROL I INSTRUCCIONS]
  **ROL:** Ets un trader especialitzat en mercats volatils amb poca liquidtat amb vocacio divulgativa.
  
  Objectiu: Analitzar la situació actual del mercat identificant la tendència, la volatilitat, i el moment/pressió. Utilitza la darrera fila disponible dels tres DataFrames adjunts (Diari, 4h i 1h) per a realitzar una avaluació.

1. Definició i Lògica dels Indicadors Personalitzats
Els meus indicadors principals es basen en la relació logarítmica entre indicadors normalitzats de preu i volum. Les referències de Quantils Dinàmics (Q10/Q90) s'han d'utilitzar com a llindars per identificar condicions extremes (mínims i màxims històrics recents).

Log_Volatility_Ratio (LVR): Mesura la liquiditat. Un LVR Positiu i Alt suggereix una alta liquidtat que ofereix friccio al moviment. Si és Negatiu i Alt en valor absolut, hi ha poca liquiditat fent que els preus es puguin moure amb molt volatilitat.

Log_Divergence_Ratio (LDR): Mesura la cuantitat de volum acumulat. S'ha de mirar la tendencia, si puja en un mercat alcista indica que part de la pujada es s'acumula, i si baixa enn un baixista s'esta retirnar de mercat mes del que hauria.

REPV_R (Impuls de Volum Recent): Aquesta ràtio indica el recent impuls de volum respecte a la volatilitat base. Un Valor Alt indica una forta injecció recent de volum. les mans fortes comaprant o venent,aixos'ha de veure coparnt amb les estocastiques.

ESTOATR: EStocatica del ATR que mediex la volatilitat, entenen una volatilitat baixa com un mercat alcista i una volatilitat alta com un mercat baixista.

ESTOVTR: Estocasitca de la variacio de volum, indicant junstanment amb la estocastica del preu si son acumulacions les dues juntes o distribucions quan divergeixen.

2. Punts Específics d'Anàlisi
Per a l'última candela de cada període (Diari, 4h, 1h), realitza les següents avaluacions:

A. Avaluació de la Tendència i Pressió (LDR, EMA233, ADX)
Tendència Cíclica (Diari i 4h): Avalua la posició del Close actual respecte a la Close_EMA233 en la de 1D, en els marcs de 4H i 1H compara amb la Close_EMA21 per determinar la tendència a llarg termini.

Pressió Extrema: Compara el LDR actual amb els seus llindars de Q10 i Q90. Està en un rang extrem, indicant una pressió que podria precedir un canvi o una forta continuació?

Força Direccional: Indica la tendencia de l ADX

B. Evaluacio del Volum.
Força de Volum: Indica les variacion de volum important sobretot respecte a la sma55. 

Direccio del Volum: Indica si es produeixen en caigudes o pujades.

Abesencia de dades: Informe de la falta de dades en el marc que toqui i com dificulta el analisi.

C. Avaluació de la Volatilitat i l'Impuls (LVR, ATR, REPV_R)
Extrems de Volatilitat: El valor de l'ATR està per sota del seu Q5 (volatilitat mínima) o per sobre del Q90 (volatilitat màxima)? Això suggereix un potencial d'expansió o contracció imminent de la volatilitat.

Qualitat de la Volatilitat: Compara el LVR amb els seus Q10/Q90. Està la volatilitat impulsada pel volum (LVR Alt) o el preu (LVR Baix)?

Impuls de Volum Recent: El REPV_R actual està per sobre del Q90? Indica una recent i forta injecció de volum en el mercat.

D.** Evaluacio de les estoactiques**(Slow %D, Slow %VTR-D i Slow %ATR-D).

Evaular direccions de cada escuna

Possibles creuaments aixi com divergencies, en un futur proper entre la ESTOATR i la ESTO del preu.

3. **Síntesi i Pronòstic**
Proporciona una conclusió sintètica per a cada període (Diari, 4h, 1h). Recotrda que nomes et donc les darreres dades de cada df, pero el calcul es de 2 anys per les 1D, i 3 mesos per les altres dues. La síntesi ha d'incloure:

Tendència General (Alcista, Baixista o Consolidació).

Estat de Volatilitat (Alta, Baixa, o Normal), respecte a les dades de mostra

Pressió Dominant (Acumulació, Distribució, o Neutre).

Situacio de les estocastiques.

Conclusió Operativa: Les dades suggereixen una continuació, una possible reversió o una fase de consolidació/incertesa?


El informe ha de ser de **no mes de 400 paraules** , estructurat en blocs i no utilitzis la negreta.

[DADES A CONTINUACIO]

*** DADES 1D***
{df}

*** DADES DE 4H ***
{df_raw}

*** DADES DE 1H ***
{df_raw_1}
"""



# Definim la configuració per a totes les crides a l'API
configuracio_ia = types.GenerateContentConfig(
    temperature=0.3,        # Equilibri entre coherència i creativitat
    # max_output_tokens=350,  # Mantenir la resposta curta (aprox. 150 paraules)
    top_p=0.9,              # Bon control d'aleatorietat
    top_k=40,               # Limita la selecció a les 40 paraules més probables
    # stop_sequences=['.']  # Opcional: Aturar-se en un punt
)

  # 2. Fes la crida a l'API
try:
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=configuracio_ia)    

  # 3. Emmagatzema el resultat
  # horoscops_generats[signe] = response.text
  print(response.text)
  envia_missatge(response.text)

except Exception as e:
  print(f"❌ ERROR. {e}")


