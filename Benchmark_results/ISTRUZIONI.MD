Istruzioni per come devono essere utilizzati i dati benchmark nella relazione:

Ogni mappa (A-B) necessita di un sotto capitolo nel quale vengono mostrati i due test principali ovvero le 32k simulazioni e le 100 x 3 seed simulazioni. Per ognuno di questi due test occore fare una media delle prestazioni come tasso di completamento e media dei tick di completamento (per il test 2 ovvero quello delle 100 x 3 seed simulazioni si può fare una media totale dei 3 seed). Per ogni Test è necessatio utilizzare ogi immagine fornita nelle apposite cartelle al fine di rendere la relazione completa. Il nome delle immagini le descrive a pieno

Ogni simulazione è dotata di una struttura del tipo:

A/
├── SIMULAZIONE              
│   ├── csv/
│   │   └── summary.csv   
│   ├── json/
│   │   ├── metadata.json  # Metadati della simulazione
│   │   ├── curves.json    # Curve dei risultati (escluso da repository)
│   │   └── results.json   # Dati dettagliati dei risultati
│   ├── Top10_presets/     # I 10 migliori preset trovati
│   │   ├── 1.Preset_23781.json
│   │   ├── 2.Preset_23783.json
│   │   └── ... (fino a 10)
│   └── Images/
│       ├── plot/          # Grafici e visualizzazioni (INCLUSI - da utilizzare nella relazione)
│       └── tabelle/       # Tabelle in formato immagine (INCLUSI - da utilizzare nella relazione)
│

Nota bene:
- Top10_presets è presente solo nei test con 32k simulazioni.
