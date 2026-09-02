import os
import time
from pathlib import Path
from openai import OpenAI

# --- CONFIGURAZIONE PERCORSI ---
DIR_ORIGINALE = "C:/Users/Fabia/Desktop/Tesi/notizieansa/notizie_referendum"
DIR_SOCIAL = "C:/Users/Fabia/Desktop/Tesi/notizieansa/notizie_social"

# --- CONFIGURAZIONE GROQ ---
# INSERISCI QUI LA TUA NUOVA CHIAVE (Non quella dello screenshot!)
GROQ_TOKEN = "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"
BASE_URL = "https://api.groq.com/openai/v1"
#MODELLO = "llama-3.1-8b-instant" # Modello ottimizzato per latenza bassissima
MODELLO = "meta-llama/llama-4-scout-17b-16e-instruct"


def riassumi_tutte_le_notizie():
    # Crea la cartella di destinazione se non esiste
    os.makedirs(DIR_SOCIAL, exist_ok=True)
    
    # Inizializza il client verso Groq
    client = OpenAI(
        api_key=GROQ_TOKEN,
        base_url=BASE_URL,
    )
    
    files = [f for f in os.listdir(DIR_ORIGINALE) if f.endswith(".txt")]
    print(f"Inizio pre-processing di {len(files)} notizie con Groq...")
    
    for i, filename in enumerate(files):
        path_in = os.path.join(DIR_ORIGINALE, filename)
        path_out = os.path.join(DIR_SOCIAL, filename)
        
        # Salta se l'abbiamo già processata in precedenza
        if os.path.exists(path_out):
            continue
            
        with open(path_in, 'r', encoding='utf-8') as f:
            testo_grezzo = f.read()
            
        prompt = f"""Sei un redattore esperto dell'agenzia stampa ANSA.
Il tuo compito è sintetizzare la seguente notizia in un singolo post social (massimo 280 caratteri).
Devi catturare il fatto politico principale in modo neutrale e distaccato. NESSUN hashtag, NESSUN commento personale.
RISPONDI SOLO ED ESCLUSIVAMENTE CON IL TESTO DEL POST SOCIAL, senza introdurlo.

NOTIZIA ORIGINALE:
{testo_grezzo}
"""
        
        try:
            print(f"[{i+1}/{len(files)}] Riassumendo {filename}...")
            
            # Chiamata al modello su Groq
            response = client.chat.completions.create(
                model=MODELLO,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, 
                max_tokens=150
            )
            
            riassunto = response.choices[0].message.content
            riassunto_pulito = riassunto.replace('"', '').strip()
            
            with open(path_out, 'w', encoding='utf-8') as f_out:
                f_out.write(riassunto_pulito)
                
            # Groq consente ~30 chiamate al minuto. 
            # Aspettiamo 2.5 secondi tra una e l'altra per stare sereni.
            time.sleep(2.5) 
            
        except Exception as e:
            print(f"Errore su {filename}: {str(e)}")
            if '429' in str(e):
                print("Rate limit raggiunto, pausa di 30 secondi...")
                time.sleep(30)

if __name__ == "__main__":
    riassumi_tutte_le_notizie()