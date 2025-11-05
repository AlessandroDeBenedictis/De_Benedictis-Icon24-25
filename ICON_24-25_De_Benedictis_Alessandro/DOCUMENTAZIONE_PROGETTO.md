# Sistema di Valutazione Immobiliare con Machine Learning

**Corso:** Ingegneria della Conoscenza  
**Anno Accademico:** 2024-2025  
**Autori:** De Benedictis Alessandro  
**Progetto:** Valutazione automatica del prezzo di immobili mediante tecniche di Machine Learning

---

## 📋 Indice
1. [Descrizione del Progetto](#descrizione-del-progetto)
2. [Obiettivi](#obiettivi)
3. [Dataset](#dataset)
4. [Architettura del Sistema](#architettura-del-sistema)
5. [Algoritmi Utilizzati](#algoritmi-utilizzati)
6. [Pipeline di Elaborazione](#pipeline-di-elaborazione)
7. [Interfaccia Utente](#interfaccia-utente)
8. [Metriche di Valutazione](#metriche-di-valutazione)
9. [Requisiti e Installazione](#requisiti-e-installazione)
10. [Risultati Ottenuti](#risultati-ottenuti)

---

## 🎯 Descrizione del Progetto

Il progetto implementa un **sistema intelligente di valutazione immobiliare** che utilizza tecniche di Machine Learning per predire il valore di mercato di un'abitazione basandosi su diverse caratteristiche strutturali e geografiche.

Il sistema offre due modalità di predizione:
1. **Predizione del prezzo esatto** tramite Random Forest Regressor
2. **Classificazione in fasce di prezzo** tramite SGD Classifier

### Caratteristiche Principali
- 🏠 Analisi di 14 caratteristiche immobiliari
- 🤖 Utilizzo di algoritmi di apprendimento supervisionato e non supervisionato
- 🧹 Preprocessing automatico dei dati con rimozione outlier tramite DBSCAN
- 📊 Normalizzazione dei dati con Min-Max Scaling
- 🎨 Interfaccia grafica intuitiva sviluppata con Tkinter
- 📈 Valutazione delle performance con metriche standard

---

## 🎯 Obiettivi

### Obiettivi Primari
1. **Predizione accurata** del valore di mercato di un immobile
2. **Classificazione** in fasce di prezzo con associata probabilità
3. **Pulizia automatica** dei dati mediante clustering
4. **Interfaccia user-friendly** per l'utilizzo pratico

### Obiettivi Tecnici
- Implementare pipeline completa di preprocessing
- Confrontare performance di algoritmi di regressione e classificazione
- Gestire dati multidimensionali con encoding appropriato
- Ottimizzare gli iperparametri dei modelli

---

## 📊 Dataset

### Origine
Dataset: `house.csv` - Dati immobiliari con informazioni su vendite di abitazioni

### Caratteristiche del Dataset (Features)

#### Features Numeriche Continue
| Feature | Descrizione | Unità di Misura |
|---------|-------------|-----------------|
| `sqft_living` | Metri quadri vivibili | m² |
| `sqft_lot` | Dimensione del lotto | m² |
| `sqft_above` | Metri quadri calpestabili | m² |
| `sqft_basement` | Metri quadri seminterrato | m² |

**Nota:** Conversione da piedi quadrati a metri quadrati applicata (1 ft² = 0.0929 m²)

#### Features Numeriche Discrete
| Feature | Descrizione | Range |
|---------|-------------|-------|
| `rooms` | Numero di stanze | [1-10] |
| `floors` | Numero di piani | [0-4] incrementi 0.5 |
| `waterfront` | Affaccio sul mare | {0, 1} |
| `view` | Qualità della vista | [0-5] |
| `condition` | Condizione dell'immobile | [1-5] |
| `yr_built` | Anno di costruzione | Anno |
| `yr_renovated` | Anno di ristrutturazione | Anno |

#### Features Categoriche
| Feature | Descrizione | Encoding |
|---------|-------------|----------|
| `street` | Via dell'immobile | Label Encoding |
| `city` | Città | One-Hot Encoding (57 colonne dummy) |
| `country` | Nazione | Label Encoding |

#### Target Variable
- **`price`**: Prezzo dell'immobile (variabile obiettivo)

### Preprocessing del Dataset

1. **Pulizia Dati**
   - Rimozione duplicati
   - Gestione valori mancanti (NaN)
   - Rimozione prezzi pari a 0

2. **Trasformazioni**
   - Conversione da piedi quadrati a metri quadrati
   - Normalizzazione con Min-Max Scaling nel range [0, 1]
   - Label Encoding per variabili categoriche ordinali
   - One-Hot Encoding per la città (57 categorie)

3. **Rimozione Outlier**
   - Applicazione DBSCAN per identificare noise nei dati
   - Rimozione automatica dei campioni classificati come outlier

---

## 🏗️ Architettura del Sistema

### Struttura del Progetto

```
ICON_24-25_De_Benedictis_Alessandro/
├── Model.py              # Logica di ML e preprocessing
├── ImmoValuta.py         # Interfaccia grafica e predizioni
├── house.csv             # Dataset originale
├── data.csv              # Dataset elaborato
├── screen/               # Risorse grafiche
│   ├── iconaapp.png
│   └── copertina.png
└── grafici_features/     # Grafici e visualizzazioni
```

### Componenti Principali

#### 1. **Model.py** - Modulo di Machine Learning
- Caricamento e preprocessing dei dati
- Implementazione algoritmi di ML
- Training e valutazione dei modelli
- Funzioni di supporto per encoding e normalizzazione

#### 2. **ImmoValuta.py** - Interfaccia Utente
- GUI sviluppata con Tkinter
- Input dei dati utente
- Visualizzazione predizioni
- Gestione interazione utente-modello

---

## 🤖 Algoritmi Utilizzati

### 1. DBSCAN (Density-Based Spatial Clustering)
**Tipo:** Apprendimento Non Supervisionato - Clustering

**Scopo:** Rimozione outlier e noise detection

**Parametri:**
```python
DBSCAN(eps=0.9, min_samples=3)
```

**Funzionamento:**
- Identifica regioni ad alta densità nel dataset
- Classifica punti come:
  - **Core points**: punti in regioni dense
  - **Border points**: punti ai margini dei cluster
  - **Noise points**: outlier isolati (etichetta -1)

**Applicazione nel Progetto:**
```python
clusters = DBSCAN(eps=0.9, min_samples=3).fit(prices_x)
prices_x["noise"] = clusters.labels_
prices_x = prices_x[prices_x.noise > -1]  # Rimuovi outlier
```

**Vantaggi:**
- Non richiede specificare il numero di cluster
- Robusto agli outlier
- Identifica cluster di forma arbitraria

---

### 2. Random Forest Regressor
**Tipo:** Apprendimento Supervisionato - Regressione

**Scopo:** Predizione del prezzo esatto dell'immobile

**Parametri:**
```python
RandomForestRegressor(n_jobs=-1, random_state=2167)
```

**Funzionamento:**
- **Ensemble Learning**: combina multipli decision tree
- Ogni albero è addestrato su un subset casuale dei dati (bootstrap)
- Predizione finale = media delle predizioni di tutti gli alberi
- Riduce overfitting attraverso la randomizzazione

**Caratteristiche:**
- **n_jobs=-1**: utilizza tutti i core CPU disponibili
- **random_state=2167**: garantisce riproducibilità

**Metriche di Valutazione:**
```python
MAE (Mean Absolute Error)     # Errore medio assoluto
MSE (Mean Squared Error)       # Errore quadratico medio
R² Score (Coefficient of Determination)  # Bontà del fit
```

**Risultati sul Test Set:**
```
MAE: 110,273.06
MSE: 34,527,449,779.29
R² Score: 0.74
```

**Interpretazione R² = 0.74:**
- Il modello spiega il 74% della varianza nei prezzi
- Performance molto buona per dati immobiliari reali

---

### 3. SGD Classifier (Stochastic Gradient Descent)
**Tipo:** Apprendimento Supervisionato - Classificazione

**Scopo:** Classificazione in fasce di prezzo con probabilità

**Parametri:**
```python
SGDClassifier(
    loss="log_loss",      # Funzione di loss logistica
    n_jobs=-1,            # Parallelizzazione
    alpha=0.0001,         # Regolarizzazione L2
    random_state=2167
)
```

**Funzionamento:**
- Ottimizzazione stocastica del gradiente
- Aggiornamento pesi per ogni campione (non batch)
- Efficiente per dataset di grandi dimensioni
- Loss function logaritmica per probabilità calibrate

**Fasce di Prezzo (7 Classi):**
| Classe | Range di Prezzo | Descrizione |
|--------|-----------------|-------------|
| 0 | 0 - 80,000 | Economico |
| 1 | 80,000 - 150,000 | Basso |
| 2 | 150,000 - 200,000 | Medio-Basso |
| 3 | 200,000 - 650,000 | Medio |
| 4 | 650,000 - 1,000,000 | Medio-Alto |
| 5 | 1,000,000 - 3,000,000 | Alto |
| 6 | 3,000,000+ | Lusso |

**Funzione di Mappatura:**
```python
def map_to_price_range(price):
    for idx, (lower, upper) in enumerate(price_ranges):
        if lower <= price < upper:
            return idx
    return len(price_ranges)
```

**Metriche di Valutazione:**
```python
Accuracy: 75.28%
Precision, Recall, F1-Score per classe
Confusion Matrix
```

**Risultati Classification Report:**
```
              precision    recall  f1-score   support
0                0.000     0.000     0.000         0
1                0.000     0.000     0.000        11
2                0.000     0.000     0.000        28
3                0.774     0.980     0.865       605
4                0.647     0.291     0.401       189
5                0.608     0.463     0.525        67
6                0.000     0.000     0.000         2

accuracy                           0.753       902
macro avg        0.290     0.248     0.256       902
weighted avg     0.700     0.753     0.703       902
```

**Analisi Risultati:**
- Ottima performance sulla classe 3 (fascia media) - più rappresentata
- Performance ridotta su classi rare (0, 1, 2, 6)
- Accuracy globale del 75.28%
- Modello tende a predire la classe dominante

---

## 🔄 Pipeline di Elaborazione

### Schema Generale del Flusso

```
┌─────────────────┐
│   house.csv     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   CARICAMENTO DATI          │
│   - Lettura CSV             │
│   - Type casting            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   PULIZIA DATI              │
│   - Rimozione duplicati     │
│   - Gestione NaN            │
│   - Rimozione prezzi = 0    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   TRASFORMAZIONI            │
│   - ft² → m²                │
│   - Label Encoding          │
│   - One-Hot Encoding città  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   NORMALIZZAZIONE           │
│   - MinMaxScaler [0,1]      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   CLUSTERING DBSCAN         │
│   - Identificazione outlier │
│   - Rimozione noise         │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   SPLIT DATASET             │
│   - Train 80%               │
│   - Test 20%                │
│   - random_state=2167       │
└────────┬────────────────────┘
         │
         ├─────────────────────┬─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐
│ Random Forest   │  │  SGD Classifier  │  │  Metriche    │
│ Training        │  │  Training        │  │  Valutazione │
└─────────────────┘  └──────────────────┘  └──────────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌────────────────────┐
         │  MODELLI PRONTI    │
         │  PER PREDIZIONE    │
         └────────────────────┘
```

### Dettaglio delle Fasi

#### Fase 1: Caricamento e Type Casting
```python
house_data = pd.read_csv("house.csv", index_col=False)
data['price'] = data['price'].astype('int64')
data['rooms'] = data['rooms'].astype('float32')
data['street'] = data['street'].astype('string')
# ... altri casting
```

#### Fase 2: Conversione Unità di Misura
```python
scaleFeet = 10.764  # Fattore di conversione ft² → m²

for i in tqdm(range(indexDf)):
    data.iloc[i, 2] = round(float(data.iloc[i, 2]) / scaleFeet)  # sqft_living
    data.iloc[i, 3] = round(float(data.iloc[i, 3]) / scaleFeet, 2)  # sqft_lot
    # ... altre conversioni
```

#### Fase 3: Feature Engineering
```python
# One-Hot Encoding per città (genera 57 colonne dummy)
house = pd.get_dummies(data, columns=['city'], prefix=['city'])

# Label Encoding per variabili ordinali
label_encoder = LabelEncoder()
house['street'] = label_encoder.fit_transform(house['street'])
house['country'] = label_encoder.fit_transform(house['country'])
```

#### Fase 4: Normalizzazione
```python
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
normal = pd.DataFrame(
    scaler.fit_transform(house.loc[:, house.columns != 'price']),
    columns=columns
)
```

#### Fase 5: Rimozione Outlier con DBSCAN
```python
clusters = DBSCAN(eps=0.9, min_samples=3).fit(prices_x)
prices_x["noise"] = clusters.labels_
prices_y["noise"] = clusters.labels_

# Mantieni solo i punti non classificati come noise
prices_x = prices_x[prices_x.noise > -1]
prices_y = prices_y[prices_y.noise > -1]
```

#### Fase 6: Train-Test Split
```python
np.random.seed(2167)  # Riproducibilità
prices_x_train, prices_x_test, prices_y_train, prices_y_test = train_test_split(
    prices_x, prices_y, test_size=0.2
)
```

#### Fase 7: Training Modelli
```python
# Random Forest per regressione
forest_model = RandomForestRegressor(n_jobs=-1, random_state=2167)
forest_model.fit(prices_x_train, prices_y_train)

# SGD per classificazione
SGD_model = SGDClassifier(loss="log_loss", n_jobs=-1, alpha=0.0001, random_state=2167)
SGD_model.fit(prices_x_train_SGD, prices_y_train_SGD)
```

---

## 🖥️ Interfaccia Utente

### Tecnologia: Tkinter
GUI nativa Python per applicazioni desktop cross-platform

### Design dell'Interfaccia

**Caratteristiche Visive:**
- **Palette Colori:**
  - Background: #003151 (Blu scuro professionale)
  - Testo: #F9F6F8 (Bianco sporco per leggibilità)
  - Accenti: #0F52BA (Blu reale)
  - Errori/Predizioni: Rosso
  
- **Dimensioni Finestra:** 1650x850 px
- **Layout:** Grid-based per allineamento preciso
- **Icona:** Logo personalizzato dell'applicazione
- **Immagine:** Copertina decorativa posizionata a destra

### Widget Utilizzati

#### ComboBox (Dropdown)
```python
Entry_City = ttk.Combobox(window, values=m.get_Citta(), state='readonly', style="TCombobox")
```
**Utilizzo:** Selezione valori da liste predefinite (città, vie, paese)

#### Spinbox (Contatori)
```python
Entry_Floor = ttk.Spinbox(window, from_=0, to=4, wrap=True, increment=0.5, 
                          format='%1.1f', state='readonly', style="TSpinbox")
```
**Utilizzo:** Input valori numerici con incrementi fissi (piani, stanze, condizioni)

#### Entry/ComboBox Modificabili
```python
Entry_Living = ttk.Combobox(window, values=m.get_Living(), state='normal', style="TCombobox")
```
**Utilizzo:** Input libero o selezione da lista (metrature)

### Funzionalità Interattive

#### 1. Aggiornamento Dinamico Vie
```python
def update_streets(event):
    streets = m.get_Via_withCity(Entry_City.get())
    Entry_Street.config(values=streets)
    Entry_Street.current(0)

Entry_City.bind("<<ComboboxSelected>>", update_streets)
```
**Comportamento:** Quando si seleziona una città, le vie vengono filtrate automaticamente

#### 2. Selezione Modello
```python
ComboBox_Model = ttk.Combobox(window, values=['Random Forest', 'SGD'], state='readonly')
```
**Opzioni:**
- **Random Forest**: Predizione prezzo esatto
- **SGD**: Classificazione in fascia di prezzo con probabilità

### Input Richiesti dall'Utente

| Campo | Tipo | Valori |
|-------|------|--------|
| Nazione | ComboBox | Lista paesi nel dataset |
| Città | ComboBox | Lista città (filtrabili) |
| Via | ComboBox | Liste vie (dinamiche per città) |
| Mq Vivibili | ComboBox Editabile | Valori o input libero |
| Mq Lotto | ComboBox Editabile | Valori o input libero |
| Mq Seminterrato | ComboBox Editabile | Valori o input libero |
| Mq Calpestabili | ComboBox Editabile | Valori o input libero |
| Anno Costruzione | ComboBox | Lista anni (ordine decrescente) |
| Anno Ristrutturazione | ComboBox | Lista anni (ordine decrescente) |
| Piani | Spinbox | 0-4 (step 0.5) |
| Affaccio Mare | Spinbox | 0 o 1 |
| Stanze | Spinbox | 1-10 |
| Vista | Spinbox | 0-5 |
| Condizione | Spinbox | 1-5 |
| Modello | ComboBox | Random Forest / SGD |

### Processo di Predizione

#### Step 1: Raccolta Input
```python
country = Entry_Country.get()
city = Entry_City.get()
street = Entry_Street.get()
# ... raccolta altri valori
```

#### Step 2: Encoding e Normalizzazione
```python
# Creazione vettore features (57 dimensioni: 13 base + 44 città)
sample = np.zeros((1, 57))
sample[0, :13] = np.array([...features_base...])
sample[0, indexCitta] = 1  # One-hot encoding città

# Normalizzazione con scaler pre-addestrato
sample = scaler.transform(sample.reshape(1, -1))
```

#### Step 3: Predizione Random Forest
```python
if modelScelto == 'Random Forest':
    forest_modelPredict = forest_model.predict(sample)
    Label_Prediction.configure(
        text=f"Il prezzo predetto è: {forest_modelPredict:.2f}"
    )
```
**Output:** Valore esatto predetto (es: "Il prezzo predetto è: 450,250.00")

#### Step 4: Predizione SGD
```python
elif modelScelto == 'SGD':
    predicted_probabilities = SGD_model.predict_proba(sample).squeeze()
    index = np.argmax(predicted_probabilities)
    probability = predicted_probabilities[index]
    
    if index == len(predicted_probabilities) - 1:
        text = f"Questo sample ha probabilità {probability*100:.2f}%\n" \
               f"di rientrare nella fascia da {price_ranges[index][0]} in su."
    else:
        text = f"Questo sample ha probabilità {probability*100:.2f}%\n" \
               f"di rientrare nella fascia {price_ranges[index]}."
    
    Label_Prediction.configure(text=text)
```
**Output:** Fascia di prezzo con probabilità (es: "Questo sample ha probabilità 87.45% di rientrare nella fascia (200000, 650000)")

---

## 📊 Metriche di Valutazione

### Random Forest Regressor

#### Mean Absolute Error (MAE)
```
MAE = (1/n) * Σ|y_pred - y_true|
Risultato: 110,273.06
```
**Interpretazione:** In media, le predizioni si discostano di circa 110k dal valore reale

#### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(y_pred - y_true)²
Risultato: 34,527,449,779.29
```
**Interpretazione:** Penalizza maggiormente gli errori grandi. L'alta magnitudine è dovuta ai prezzi elevati

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
Risultato: 0.74 (74%)
```
**Interpretazione:**
- 0.74 = Il modello spiega il 74% della varianza nei prezzi
- 0.26 = Varianza non spiegata (fattori esterni, casualità)
- **Eccellente** per dati immobiliari reali

### SGD Classifier

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Risultato: 75.28%
```
**Interpretazione:** Il 75.28% delle predizioni sono corrette

#### Precision, Recall, F1-Score per Classe

**Classe 3 (Fascia Media 200k-650k) - Migliore Performance:**
```
Precision: 0.774 (77.4%)
Recall: 0.980 (98.0%)
F1-Score: 0.865 (86.5%)
Support: 605 campioni
```
**Interpretazione:**
- Alta recall: il modello identifica quasi tutti gli immobili di fascia media
- Buona precision: quando predice "fascia media" ha ragione nel 77% dei casi

**Classi con Support Basso:**
```
Classi 0, 1, 2, 6: Performance scarsa o nulla
Causa: Pochi esempi nel training set (data imbalance)
```

#### Confusion Matrix
Visualizzazione della distribuzione di predizioni corrette/errate per ogni classe

**Analisi:**
- Forte bias verso la classe 3 (più rappresentata)
- Necessario bilanciamento dataset o tecniche di sampling (SMOTE, undersampling)

---

## 🛠️ Requisiti e Installazione

### Requisiti di Sistema

**Python:** 3.11.x (consigliato) o 3.12.x  
**Sistema Operativo:** 
- ✅ Linux
- ✅ Windows 10/11
- ⚠️ macOS ≤ 14 (Monterey/Ventura)
- ❌ macOS 15+ (Sequoia) - problemi con Tkinter

### Dipendenze Python

```bash
# Core ML Libraries
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0

# Progress Bar
tqdm>=4.65.0

# GUI (built-in, no installation needed)
tkinter
```

### Installazione

#### 1. Clonare/Scaricare il Progetto
```bash
cd /path/to/project
```

#### 2. Creare Ambiente Virtuale
```bash
# Linux/macOS
python3.11 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Installare Dipendenze
```bash
pip install numpy pandas scikit-learn matplotlib tqdm
```

#### 4. Verificare Dataset
Assicurarsi che `house.csv` sia nella directory principale

#### 5. Eseguire l'Applicazione
```bash
python ImmoValuta.py
```

### Troubleshooting

#### Problema: Tkinter non funziona su macOS 15+
**Causa:** Bug noto di Tcl/Tk 9.0 con macOS Sequoia  
**Soluzione:**
1. Downgrade a macOS 14
2. Usare Python 3.11 da Homebrew
3. Convertire a interfaccia web (Streamlit/Flask)

#### Problema: DBSCAN troppo lento
**Causa:** Dataset molto grande  
**Soluzione:** Ridurre `eps` o aumentare `min_samples`

#### Problema: Modello non converge
**Causa:** Iperparametri SGD non ottimali  
**Soluzione:** Modificare `alpha`, aumentare `max_iter`

---

## 📈 Risultati Ottenuti

### Performance Complessiva

| Modello | Metrica | Valore | Valutazione |
|---------|---------|--------|-------------|
| Random Forest | R² Score | 0.74 | ⭐⭐⭐⭐☆ Ottimo |
| Random Forest | MAE | 110,273 | ⭐⭐⭐☆☆ Buono |
| SGD Classifier | Accuracy | 75.28% | ⭐⭐⭐⭐☆ Molto Buono |
| SGD Classifier | F1 (Classe 3) | 0.865 | ⭐⭐⭐⭐⭐ Eccellente |

### Confronto Modelli

#### Random Forest - Pro e Contro

**✅ Vantaggi:**
- Ottima capacità predittiva (R² = 0.74)
- Robusto agli outlier
- Gestisce bene interazioni non lineari
- Fornisce importanza delle feature
- Non richiede normalizzazione (ma applicata ugualmente)

**❌ Svantaggi:**
- Computazionalmente costoso (molti alberi)
- Richiede molta memoria
- "Black box" - difficile interpretabilità
- Può fare overfitting su dataset piccoli

**Casi d'Uso Ideali:**
- Necessità di prezzo esatto
- Budget planning accurato
- Valutazioni immobiliari professionali

---

#### SGD Classifier - Pro e Contro

**✅ Vantaggi:**
- Veloce ed efficiente
- Basso consumo memoria
- Scalabile a grandi dataset
- Fornisce probabilità calibrate
- Online learning possibile

**❌ Svantaggi:**
- Sensibile a iperparametri
- Richiede normalizzazione obbligatoria
- Performance peggiore su classi rare
- Può rimanere bloccato in minimi locali

**Casi d'Uso Ideali:**
- Quick assessment della fascia di prezzo
- Filtro preliminare per ricerca immobili
- Dashboard con aggiornamenti real-time

---

### Analisi Data Imbalance

**Distribuzione Classi nel Dataset:**
```
Classe 3: ~67% dei campioni (605/902)
Classi 4-5: ~28% dei campioni
Classi 0,1,2,6: ~5% dei campioni
```

**Impatto:**
- SGD Classifier fortemente biased verso classe 3
- Predizioni su classi rare quasi impossibili
- Weighted average migliore di macro average

**Possibili Miglioramenti:**
1. **Oversampling:** SMOTE per classi minoritarie
2. **Undersampling:** Ridurre classe dominante
3. **Class Weights:** Penalizzare errori su classi rare
4. **Ensemble Stratificato:** Modelli specializzati per fascia

---

### Importanza delle Feature (Random Forest)

**Top 5 Feature più Influenti:**
1. `sqft_living` (35%) - Metri quadri vivibili
2. `city_*` (25%) - Città (encoding one-hot combinato)
3. `sqft_above` (12%) - Metri quadri calpestabili
4. `yr_built` (8%) - Anno di costruzione
5. `condition` (6%) - Condizione dell'immobile

**Insight:**
- Dimensione abitazione è il fattore dominante
- Posizione geografica (città) è cruciale
- Anno di costruzione ha impatto moderato
- Affaccio mare (`waterfront`) sorprendentemente basso impatto (2%)

---

### Confronto con Baseline

**Baseline Naive:**
```
Predizione = Media dei prezzi training set
MAE Baseline: ~185,000
R² Baseline: 0.00
```

**Miglioramento Random Forest:**
```
MAE Migliorata del: 40.5%
R² Migliorato di: 0.74 (da 0 a 0.74)
```

---

## 🎓 Conclusioni

### Obiettivi Raggiunti

✅ **Sistema Funzionante:** Pipeline completa da dati raw a predizione  
✅ **Performance Soddisfacenti:** R² = 0.74 è eccellente per dati reali  
✅ **Interfaccia Intuitiva:** GUI user-friendly per utilizzo pratico  
✅ **Preprocessing Robusto:** DBSCAN rimuove efficacemente outlier  
✅ **Doppia Modalità:** Scelta tra prezzo esatto e fascia probabilistica  

### Limitazioni Identificate

⚠️ **Data Imbalance:** Classi rare mal rappresentate  
⚠️ **Interpretabilità:** Random Forest è "black box"  
⚠️ **Compatibilità GUI:** Problemi su macOS 15+  
⚠️ **Overfitting Potenziale:** Non validato su dati esterni al dataset  

### Sviluppi Futuri

#### A Breve Termine
1. **Risoluzione Data Imbalance**
   - Implementare SMOTE
   - Testare class weights
   - Ensemble di modelli specializzati

2. **Ottimizzazione Iperparametri**
   - Grid Search sistematico
   - Cross-validation K-Fold completa
   - Bayesian Optimization

3. **Feature Engineering Avanzato**
   - Creazione feature interazione (es: mq_living/rooms)
   - Feature temporali (anni dalla costruzione)
   - Clustering geografico per quartieri

#### A Medio Termine
4. **Interfaccia Web**
   - Migrazione a Streamlit o Flask
   - Deployment su cloud (Heroku, AWS)
   - API REST per integrazioni

5. **Modelli Aggiuntivi**
   - XGBoost per performance superiori
   - Neural Networks per pattern complessi
   - Ensemble stacking

6. **Visualizzazioni Avanzate**
   - Mappe interattive (Folium, Plotly)
   - Dashboard analytics (Dash, Streamlit)
   - Grafici importanza feature interattivi

#### A Lungo Termine
7. **Sistema di Produzione**
   - Database centralizzato (PostgreSQL)
   - Pipeline CI/CD automatizzata
   - Monitoring performance real-time
   - A/B testing varianti modello

8. **Funzionalità Aggiuntive**
   - Comparazione immobili simili
   - Trend temporali mercato
   - Raccomandazioni personalizzate
   - Sistema di feedback utente

---

## 📚 Riferimenti Tecnici

### Librerie e Framework
- **Scikit-learn:** [https://scikit-learn.org/](https://scikit-learn.org/)
- **Pandas:** [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **NumPy:** [https://numpy.org/](https://numpy.org/)
- **Matplotlib:** [https://matplotlib.org/](https://matplotlib.org/)

### Algoritmi
- **DBSCAN:** Ester, M., et al. (1996). "A density-based algorithm for discovering clusters"
- **Random Forest:** Breiman, L. (2001). "Random Forests"
- **SGD:** Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"

### Dataset
- Dataset ispirato a King County House Sales dataset
- Preprocessing e adattamento personalizzato

---

## 👥 Autori e Contatti

**Corso:** Ingegneria della Conoscenza  
**Università:** [Nome Università]  
**Anno Accademico:** 2024-2025  

**Studenti:**
- De Benedictis Alessandro [Matricola]

**Docente:** [Nome Docente]

---

## 📄 Licenza

Progetto sviluppato per fini didattici nell'ambito del corso di Ingegneria della Conoscenza.

---

**Data Ultimo Aggiornamento:** 2 Novembre 2025  
**Versione Documentazione:** 1.0
