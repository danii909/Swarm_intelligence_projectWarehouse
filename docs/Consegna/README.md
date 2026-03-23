# Guida rapida: uso di `A.json` e `B.json` (esame 26/02/2026)

## 1) Cosa contengono i file

Ogni file (`A.json`, `B.json`) ha questa struttura:

- `metadata`: informazioni globali (`grid_size`, numero magazzini, numero oggetti)
- `grid`: matrice `N x N` di interi (mappa)
- `warehouses`: lista dei 4 magazzini con ingresso/uscita
- `objects`: lista delle coordinate degli oggetti `[riga, colonna]`

## 2) Significato delle costanti `0 1 2 3 4`

I valori nella matrice `grid[r][c]` sono:

| Valore | Nome | Significato pratico |
|---|---|---|
| `0` | `EMPTY` | Corridoio libero (percorribile) |
| `1` | `WALL` | Ostacolo/scaffale (non percorribile) |
| `2` | `WAREHOUSE` | Interno magazzino |
| `3` | `ENTRANCE` | Ingresso magazzino |
| `4` | `EXIT` | Uscita magazzino |

Coordinate sempre in formato `[riga, colonna]`, con origine in alto a sinistra (`[0,0]`).

## 3) Punto importante sugli oggetti

Gli oggetti **non** sono codificati come valore `5` nella `grid`.
La loro posizione e nel campo separato:

- `objects = [[r1, c1], [r2, c2], ...]`

Nel testo d'esame (`20260226-progetto.pdf`) la posizione degli oggetti non è nota agli agenti: quindi `objects` va trattato come *ground truth* della simulazione (o per valutazione/debug), non come informazione da dare direttamente agli agenti in fase di esplorazione.

Quando un oggetto viene portato all'interno di un magazzino, può essere considerato **consegnato** e quindi **rimosso dall'ambiente**. non resta come entità attiva nella griglia e non si sovrappone ad altri oggetti.

## 4) Esempio minimo in Python

```python
import json

EMPTY, WALL, WAREHOUSE, ENTRANCE, EXIT = 0, 1, 2, 3, 4

with open("A.json", "r", encoding="utf-8") as f:
    env = json.load(f)

grid = env["grid"]
warehouses = env["warehouses"]
objects_truth = set(map(tuple, env["objects"]))  # solo riferimento simulatore

def is_walkable(r, c):
    return grid[r][c] in (EMPTY, ENTRANCE, EXIT) # Assicurarsi che ad esempio l'entrata e l'uscita siano percorribili solo in un senso (in base da dove viene l'agente)
```

## 5) Differenza tra istanza A e B

- formato identico
- stessi metadati (`25x25`, `4` magazzini, `10` oggetti)
- cambia la disposizione della mappa (`grid`) e la posizione degli oggetti (`objects`)

## 6) Visualizzazione rapida

Per controllare visivamente un'istanza:

```bash
python visualize_environment.py A.json A.png
python visualize_environment.py B.json B.png
```

## 7) Caratteristiche agente

### Percezione

| Parametro | Intervallo | Note |
|---|---|---|
| Raggio di visibilità | `[1, 3]` celle | Determina quante celle intorno a sé l'agente può osservare (Manhattan con occlusione). Con raggio 1 vede solo le 4 celle adiacenti; con raggio 3 ha una visione più ampia. |
| Raggio di comunicazione | `[1, 2]` celle | Distanza massima entro cui due agenti possono scambiarsi informazioni (es. mappa esplorata, posizione oggetti trovati). |

### Popolazione e strategie

- **Numero di agenti:** 5
- Ogni agente dovrebbe adottare una **strategia di ricerca diversa** per confrontare l'efficacia e favorire la copertura collaborativa della mappa.

### Energia (batteria)

| Parametro | Valore |
|---|---|
| Batteria iniziale | `500` unità |
| Consumo per step | `1` unità per mossa |
| Batteria esaurita | L'agente si ferma e non può più muoversi |

> **Nota:** la batteria è individuale per agente. Con 500 di batteria e consumo 1/step, ogni agente può fare al massimo 500 mosse prima di fermarsi. Considerare implementare strategie di risparmio energetico o rientro al magazzino in caso di batteria bassa.

## 8) Suggerimenti per il debugging

### Visualizzazione in tempo reale

Durante lo sviluppo è utile avere una **visualizzazione live** dell'ambiente e del comportamento degli agenti. Alcune opzioni:

| Approccio | Libreria | Pro |
|---|---|---|
| Plot aggiornato a ogni tick | `matplotlib` + `plt.pause()` | Semplice, già usata nel progetto |
| Animazione esportabile | `matplotlib.animation` | Genera GIF/MP4 riproducibili |
| Interfaccia interattiva | `pygame` | Fluida, permette controlli (pausa, step-by-step) |

### Esempio rapido con Matplotlib (ESEMPI DI CODICE)

```python
import matplotlib.pyplot as plt
import time

fig, ax = plt.subplots(figsize=(8, 8))
plt.ion()  # modalità interattiva

for tick in range(max_ticks):
    ax.clear()
    # disegna griglia, magazzini, oggetti...
    for agent in agents:
        ax.plot(agent.col, agent.row, "o", markersize=8, label=agent.name)
    ax.set_title(f"Tick {tick}")
    ax.legend(loc="upper right", fontsize=7)
    plt.pause(0.05)

plt.ioff()
plt.show()
```

### Logging strutturato (ESEMPI DI CODICE)

Oltre alla visualizzazione, salvare un **log JSON** tick-per-tick facilita l'analisi post-simulazione:

```python
log = []
for tick in range(max_ticks):
    snapshot = {
        "tick": tick,
        "agents": [
            {
                "id": a.id,
                "pos": [a.row, a.col],
                "battery": a.battery,
                "carrying": a.carrying,
                "strategy": a.strategy_name,
            }
            for a in agents
        ],
        "objects_remaining": len(objects_not_delivered),
    }
    log.append(snapshot)

# salva alla fine
import json
with open("simulation_log.json", "w") as f:
    json.dump(log, f, indent=2)
```
