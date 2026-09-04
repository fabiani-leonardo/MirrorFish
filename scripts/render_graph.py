#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import networkx as nx
from pyvis.network import Network

def generate_graph(json_path, output_path):
    print(f"Caricamento dati da {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Inizializza il grafo NetworkX
    G = nx.DiGraph()

    # Aggiungi i nodi (Agenti)
    for node in data.get('nodes', []):
        agent_id = node.get('id')
        username = node.get('username', f"Agent_{agent_id}")
        
        # Colora in base all'orientamento politico (se disponibile nel JSON)
        # Puoi mappare altri attributi come l'età se li hai salvati nel JSON
        color = "#97c2fc" # Blu default
        if "SI" in str(node): color = "#2f6b4f"
        elif "NO" in str(node): color = "#8f2f2f"
        
        G.add_node(agent_id, label=username, title=f"ID: {agent_id}\nUtente: {username}", color=color)

    # Aggiungi gli archi (Follows/Interactions)
    for edge in data.get('links', []):
        source = edge.get('source')
        target = edge.get('target')
        rel_type = edge.get('type', 'follows')
        
        G.add_edge(source, target, title=rel_type)

    print(f"Grafo costruito: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi.")

    # Converti per Pyvis
    net = Network(height="800px", width="100%", bgcolor="#faf8f4", font_color="#1c1a17", directed=True)
    net.from_nx(G)
    
    # Aggiunge i controlli fisici nella pagina HTML
    net.show_buttons(filter_=['physics'])
    
    net.write_html(str(output_path))
    print(f"Grafo interattivo salvato in: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera un grafo HTML da graph.json")
    ap.add_argument("json_file", help="Percorso al file graph.json")
    ap.add_argument("-o", "--out", default=None, help="Percorso di output (HTML)")
    args = ap.parse_args()

    input_path = Path(args.json_file)
    output_path = Path(args.out) if args.out else input_path.with_suffix('.html')
    
    generate_graph(input_path, output_path)