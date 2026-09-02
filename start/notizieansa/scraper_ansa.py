import os
import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

# --- CONFIGURAZIONI ---
CARTELLA_OUTPUT = "notizie_referendum"
URL_RICERCA_BASE = "https://www.ansa.it/ricerca/ansait/search.shtml?start={}&tag=&any=referendum+giustizia&sezione=&periodo=&sort=data%3Adesc"
MAX_PAGINE = 150 # Aumentato, tanto lo script si ferma da solo quando esce dal range di date

# INSERISCI QUI LE DATE ESATTE (Anno, Mese, Giorno)
DATA_FINE = datetime(2026, 3, 21)   # Esempio: Giorno del referendum (o il giorno dopo)
DATA_INIZIO = datetime(2025, 10, 30) # Esempio: Giorno dell'approvazione in Senato

MESI_IT = {
    '01': 'gennaio', '02': 'febbraio', '03': 'marzo', '04': 'aprile',
    '05': 'maggio', '06': 'giugno', '07': 'luglio', '08': 'agosto',
    '09': 'settembre', '10': 'ottobre', '11': 'novembre', '12': 'dicembre'
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def setup_cartella():
    if not os.path.exists(CARTELLA_OUTPUT):
        os.makedirs(CARTELLA_OUTPUT)

def estrai_data_da_url(url):
    """Estrae la data dall'URL e restituisce sia l'oggetto datetime che la stringa per il nome file."""
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
    if match:
        anno, mese, giorno = match.groups()
        data_obj = datetime(int(anno), int(mese), int(giorno))
        nome_mese = MESI_IT.get(mese, mese)
        data_stringa = f"{giorno}{nome_mese}{anno}"
        return data_obj, data_stringa
    return None, "data_sconosciuta"

def scarica_articolo(url, id_notizia, data_stringa):
    """Scarica la singola notizia e la salva in un file txt."""
    if not url.startswith("http"):
        url = "https://www.ansa.it" + url

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        titolo_tag = soup.find('h1')
        titolo = titolo_tag.get_text(strip=True) if titolo_tag else "Titolo non trovato"

        div_testo = soup.find('div', class_='news-txt') or soup.find('div', itemprop='articleBody')
        if div_testo:
            paragrafi = div_testo.find_all('p')
        else:
            paragrafi = soup.find_all('p')
            
        testo_pulito = "\n\n".join([p.get_text(strip=True) for p in paragrafi if len(p.get_text(strip=True)) > 40])

        nome_file = f"{data_stringa}-{id_notizia}.txt"
        percorso_file = os.path.join(CARTELLA_OUTPUT, nome_file)

        with open(percorso_file, 'w', encoding='utf-8') as f:
            f.write(f"TITOLO: {titolo}\n")
            f.write(f"DATA ESTRATTA: {data_stringa}\n")
            f.write(f"LINK: {url}\n")
            f.write("-" * 50 + "\n\n")
            f.write(testo_pulito)
            
        return True

    except Exception as e:
        print(f"  [!] Errore durante il download dell'articolo: {e}")
        return False

def avvia_scraping():
    setup_cartella()
    id_notizia_globale = 1
    
    for pagina in range(MAX_PAGINE):
        start_param = pagina * 12
        url_ricerca = URL_RICERCA_BASE.format(start_param)
        print(f"\nAnalizzando pagina {pagina + 1}...")
        
        try:
            response = requests.get(url_ricerca, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links_articoli = []
            
            # --- MODIFICA: Restringiamo il campo! ---
            # In genere i siti di news come ANSA racchiudono ogni risultato 
            # all'interno di un tag <article>, oppure dentro i titoli (es. h3).
            # Proviamo a estrarre il link SOLO se fa parte di un'intestazione o di un articolo.
            
            # Metodo: cerchiamo tutti i tag <article> o le intestazioni principali
            for blocco in soup.find_all(['article', 'h3']): 
                a = blocco.find('a', href=True)
                if a:
                    href = a['href']
                    if href.endswith('.html') and '/notizie/' in href:
                        if href not in links_articoli:
                            links_articoli.append(href)
                            
            # Se la lista è vuota, significa che la struttura HTML è diversa.
            # In quel caso dovrai fare "Tasto Destro -> Ispeziona" sulla pagina ANSA 
            # per trovare la classe esatta del contenitore dei risultati (es. div class="search-result")
            # ----------------------------------------
            
            if not links_articoli:
                print("Nessun altro articolo trovato o la struttura della pagina è cambiata.")
                break
                
            for link in links_articoli:
                data_obj, data_stringa = estrai_data_da_url(link)
                
                # Se l'URL non ha una data chiara, saltalo per sicurezza
                if not data_obj:
                    continue

                # 1. Se la notizia è più recente del giorno del referendum, saltala ma continua a cercare
                if data_obj > DATA_FINE:
                    print(f"  -> Salto: {data_stringa} (Successiva al referendum)")
                    continue
                
                # 2. Se la notizia è più vecchia dell'inizio, FERMA TUTTO
                if data_obj < DATA_INIZIO:
                    print(f"\n[!] Raggiunta una notizia del {data_stringa}. È precedente all'annuncio ({DATA_INIZIO.strftime('%d/%m/%Y')}).")
                    print("Fine automatica dello scraping!")
                    return # Esce completamente dalla funzione
                
                # 3. Se siamo qui, la notizia è nel range giusto! La scarichiamo.
                print(f"  -> Scaricando articolo del {data_stringa}: {link}")
                scarica_articolo(link, id_notizia_globale, data_stringa)
                id_notizia_globale += 1
                time.sleep(1.5) 
                
        except Exception as e:
            print(f"Errore nella pagina di ricerca: {e}")
            break

if __name__ == "__main__":
    print(f"Inizio scraping... Cerco notizie dal {DATA_INIZIO.strftime('%d/%m/%Y')} al {DATA_FINE.strftime('%d/%m/%Y')}")
    avvia_scraping()
    print("\nLavoro completato! I tuoi dati per la tesi sono nella cartella:", CARTELLA_OUTPUT)