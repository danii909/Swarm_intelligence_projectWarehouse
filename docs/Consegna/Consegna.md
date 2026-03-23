Sistema Multi-Agente per il Recupero e Sistemazioni di Oggetti
in una Rete di Magazzini

Descrizione:
Sviluppare un modello di simulazione multi-agente che coordini una flotta di agenti autonomi
incaricati di recuperare un insieme di oggetti dispersi in una vasta area logistica composta
da più magazzini, e riportarli/posizionarli in un magazzino.

Input:
- Mappa dell’ambiente
- Posizione e struttura dei magazzini
- Numero totale degli oggetti da recuperare

Nota:
La posizione degli oggetti NON è nota → gli agenti devono:
- esplorare
- rilevare
- recuperare

Obiettivo (GOAL):
Ottimizzare:
1) numero di oggetti rilevati e correttamente posizionati
2) tempo totale di recupero
3) energia media consumata

--------------------------------------------------

Struttura dell’Ambiente:

Ambiente = griglia 2D composta da:
- area esterna (corridoi, piazzali, zone di transito)
- N magazzini (blu)
  - ingressi (verde)
  - uscite (rosso)
- scaffali/muri (ostacoli, nero)
- zone possibili oggetti (giallo)

Proprietà:
- ostacoli e magazzini sono statici
- gli agenti percepiscono solo nel loro raggio visivo

--------------------------------------------------

Agente - Caratteristiche:

Ogni agente possiede:

1) posizione
2) sensore di visibilità:
   - raggio di percezione
   - rilevamento ostacoli
   - rilevamento oggetti
   - rilevamento altri agenti (anti-collisione)
   - rilevamento ingressi/uscite magazzino

3) livello di energia
4) memoria locale (mappa esplorata)
5) sensore di connessione
6) strategia esplorativa
7) capacità di decision-making

Estensioni possibili:
- agenti con ruoli diversi (es: scout, coordinator)
- strategie esplorative personalizzate
- decision-making basato su regole/euristiche

Nota sviluppo:
- progetto individuale → sovrapposizione agenti consentita
- progetto di gruppo → sovrapposizione NON consentita (salvo casi estremi)

--------------------------------------------------

Comunicazione tra agenti:

- avviene quando i raggi di comunicazione si intersecano
- scambio informazioni:
  - mappe locali
  - oggetti rilevati ma non recuperati

Possibili estensioni:
- comunicazione agente-ambiente
- comunicazione tra ruoli diversi

--------------------------------------------------

Obiettivo finale:

Ogni agente (partenza: [0,0]) deve:
- recuperare tutti gli N oggetti
- portarli nel magazzino

Ottimizzando:
- tempo totale
- distanza percorsa
- consumo energetico medio