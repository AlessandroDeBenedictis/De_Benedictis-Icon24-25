# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # Usa backend non-GUI per evitare conflitti con Tkinter
import tkinter as tk
import warnings
import numpy as np
import Model as m
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

warnings.filterwarnings('ignore')

#Funzione per il pulsante di predizione
def predizione_prezzo():

    Label_Prediction.configure(text="")
    
    country = Entry_Country.get()
    index = m.model.dataframe_UI.index[m.model.dataframe_UI['country']==country].tolist()
    country = m.model.dataframe.iloc[index[0],11] # 11 = Country
    street = Entry_Street.get()
    index = m.model.dataframe_UI.index[m.model.dataframe_UI['street']==street].tolist()
    street = m.model.dataframe.iloc[index[0],10] # 10 = Street

    city = Entry_City.get()
    indexCitta = m.model.dataframe.columns.get_loc(F"city_{city}")

    Mq_living = float((Entry_Living.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['Mq_living']==Mq_living].tolist()
    #Mq_living = m.model.dataframe.iloc[index[0],0] # 0 = Mq_living

    sqft_lot = float((Entry_Lot.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_lot']==sqft_lot].tolist()
    #sqft_lot = m.model.dataframe.iloc[index[0],1] # 1 = sqft_lot

    sqft_basement = float((Entry_Basement.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_basement']==sqft_basement].tolist()
    #sqft_basement = m.model.dataframe.iloc[index[0],7] # 7 = sqft_basement

    sqft_above = float((Entry_Above.get()))
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['sqft_above']==sqft_above].tolist()
    #sqft_above = m.model.dataframe.iloc[index[0],6] # 6 = sqft_above

    rooms = float(Entry_Room.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['rooms']==rooms].tolist()
    #rooms = m.model.dataframe.iloc[index[0],12] # 12 = rooms

    floors = float(Entry_Floor.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['floors']==floors].tolist()
    #floors = m.model.dataframe.iloc[index[0],2] # 2 = floors

    waterfront = int(Entry_WF.get())  
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['waterfront']==waterfront].tolist()
    #waterfront = m.model.dataframe.iloc[index[0],3] # 3 = waterfront

    view = round(float((Entry_View.get())))  
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['view']==view].tolist()
    #view = m.model.dataframe.iloc[index[0],4] # 4 = view

    condition = round(float((Entry_Cond.get())))  
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['condition']==condition].tolist()
    #condition = m.model.dataframe.iloc[index[0],5] # 5 = condition

    yr_built = int(Entry_YearC.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['yr_built']==yr_built].tolist()
    #yr_built = m.model.dataframe.iloc[index[0],8] # 8 = yr_built

    yr_renovated = int(Entry_YearR.get())
    #index = m.model.dataframe_UI.index[m.model.dataframe_UI['yr_renovated']==yr_renovated].tolist()
    #yr_renovated = m.model.dataframe.iloc[index[0],9] # 9 = yr_renovated
    
    #Predizione del prezzo
    sample = np.zeros((1,57))
    sample[0,:13] = np.array([Mq_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,country,rooms]).reshape(1, -1)
    sample[0,indexCitta] = 1

    sample = scaler.transform(sample.reshape(1, -1))

    modelScelto = str(ComboBox_Model.get())
    if(modelScelto == 'Random Forest'):
        forest_modelPredict = forest_model.predict(sample)
        Label_Prediction.configure(text=("Il prezzo predetto è: %.2f" %forest_modelPredict))
    elif (modelScelto == 'SGD'):

        predicted_probabilities = SGD_model.predict_proba(sample).squeeze()

        # Ritrovo l'indice della probabilità maggiore
        index = np.argmax(predicted_probabilities) # Indice della probabilità maggiore            
        probability = predicted_probabilities[index]  # Mi prendo la percentuale di probabilità in base all'indice qui sopra

        if(index == len(predicted_probabilities)-1):
            text = (F"Questo sample ha probabilità {probability*100:.2f}%\ndi rientrare nella fascia da {(price_ranges[int(index)])[0]} in su.")
        else:
            text = (F"Questo sample ha probabilità {probability*100:.2f}%\ndi rientrare nella fascia {price_ranges[int(index)]}.")
        Label_Prediction.configure(text=text)
    else:
        raise NotImplementedError(F"Scegli un modello valido. Hai scelto {modelScelto}.")

def update_streets(event):
    streets = m.get_Via_withCity(Entry_City.get())
    Entry_Street.config(values=streets)
    Entry_Street.current(0)
    return

price_ranges = [
    (0, 80000),
    (80000, 150000),
    (150000, 200000),
    (200000, 650000),
    (650000, 1000000),
    (1000000, 3000000),
    (3000000, float('inf'))
]

prices_x_train, prices_x_test, prices_y_train, prices_y_test, scaler  = m.crea_basedati(modelUsed="RandomForest")
prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD, scaler_SGD  = m.crea_basedati(modelUsed="SGD")

forest_model = m.modello(prices_x_train, prices_x_test, prices_y_train, prices_y_test)
SGD_model = m.modello2(prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD)

window = tk.Tk()


window.geometry("900x810+50+50")
window.title("ImmoValuta Pro - Sistema di Valutazione Immobiliare")
# Salta l'icona per ora - non è essenziale
window.resizable(False, False)
# Nuovo schema colori: Verde moderno
window.config(bg="#0d1b0d")

# Configura il grid per centrare tutto
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(3, weight=1)

style = ttk.Style()
style.theme_use("clam")  # Usa un tema compatibile con le modifiche
style.configure("TCombobox",
                fieldbackground="#1a2f1a",  # Sfondo della casella di testo
                background="#1a2f1a",   # Sfondo del menu a discesa
                foreground="#eee",       # Colore del testo
                arrowcolor="#00ff41",        # Colore della freccia - verde elettrico
                bordercolor="#00ff41",       # Colore del bordo
                lightcolor="#1a2f1a")   # Colore evidenziato

style.map("TCombobox",
          fieldbackground=[("readonly", "#1a2f1a")],  # Forza lo sfondo anche in readonly
          foreground=[("readonly", "#eee")],        # Colore del testo in readonly
          arrowcolor=[("readonly", "#00ff41")],        # Colore della freccia in readonly
          background=[("readonly", "#1a2f1a")],      # Sfondo del menu dropdown
          bordercolor=[("readonly","#00ff41")],       # Colore del bordo
          lightcolor=[("readonly","#1a2f1a")])   # Colore evidenziato



style.theme_use("clam")  # Usa un tema compatibile con le personalizzazioni
style.configure("TSpinbox",
                fieldbackground="#1a2f1a",
                background="#1a2f1a",   # Sfondo dello Spinbox
                foreground="#eee",       # Colore del testo
                arrowcolor="#00ff41",        # Colore delle frecce
                bordercolor="#00ff41",       # Colore del bordo
                lightcolor="#1a2f1a")   # Colore evidenziato

style.map("TSpinbox",
          fieldbackground=[("readonly", "#1a2f1a")],  # Forza lo sfondo anche in readonly
          foreground=[("readonly", "#eee")],        # Colore del testo in readonly
          arrowcolor=[("readonly", "#00ff41")],        # Colore della freccia in readonly
          background=[("readonly", "#1a2f1a")],      # Sfondo del menu dropdown
          bordercolor=[("readonly","#00ff41")],       # Colore del bordo
          lightcolor=[("readonly","#1a2f1a")])   # Colore evidenziato


# Header principale con design moderno
Label_Titolo = tk.Label(window, text="Inserisci i Dettagli dell'Immobile", 
                        font=("Segoe UI", 22, "bold"),
                        bg="#0d1b0d", 
                        fg="#00ff41")  # Verde elettrico
Label_Titolo.grid(row=0, column=0, columnspan=4, pady=(15,0))

# Sottotitolo
Label_Sottotitolo = tk.Label(window, text="Utilizza i nostri algoritmi avanzati per una valutazione precisa", 
                             font=("Segoe UI", 11),
                             bg="#0d1b0d", 
                             fg="#a8a8a8")
Label_Sottotitolo.grid(row=1, column=0, columnspan=4, pady=(5,15))

# Label per l'autore - Design premium
Label_Autore = tk.Label(window, text="Developed by De Benedictis Alessandro | AI-Powered Evaluation", 
                        font=("Consolas", 10, "bold"), 
                        bg="#1a4d1a", 
                        fg="#00ff41", 
                        padx=20, 
                        pady=8, 
                        relief="ridge", 
                        borderwidth=2)
Label_Autore.place(relx=0.5, rely=0.97, anchor="s")

#Creo l'etichetta e la combobox per il valore "Nazione"
Label_Country = tk.Label(window, text = "Nazione: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Country.grid(row=3, column=1, padx=5, pady=4, sticky="E")
Entry_Country = ttk.Combobox(window, values = m.get_Regione(), state = 'readonly', style="TCombobox", width=25)
Entry_Country.grid(row=3, column=2, padx=5, pady=4, sticky="W")
Entry_Country.current(0)

#Creo l'etichetta e la combobox per il valore "Città"
Label_City = tk.Label(window, text = "Città: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_City.grid(row=4, column=1, padx=5, pady=4, sticky="E")
Entry_City = ttk.Combobox(window, values = m.get_Citta(), state = 'readonly', style="TCombobox", width=25)
Entry_City.grid(row=4, column=2, padx=5, pady=4, sticky="W")
Entry_City.current(0)

#Creo l'etichetta e la combobox per il valore "Via"
Label_Street = tk.Label(window, text = "Via: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Street.grid(row=5, column=1, padx=5, pady=4, sticky="E")
Entry_Street = ttk.Combobox(window, values = m.get_Via(), state = 'readonly', style="TCombobox", width=25)
Entry_Street.grid(row=5, column=2, padx=5, pady=4, sticky="W")
Entry_Street.current(0)
Entry_City.bind("<<ComboboxSelected>>", update_streets) # Quando si seleziona la città,si aggiorna la lista delle strade

#Creo l'etichetta e la combobox per il valore "Mq_Vivibili"
Label_Living = tk.Label(window, text = "Metri quadri Vivibili: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Living.grid(row=6, column=1, padx=5, pady=4, sticky="E")
Entry_Living = ttk.Combobox(window, values = m.get_Living(), state = 'normal', style="TCombobox", width=25)
Entry_Living.grid(row=6, column=2, padx=5, pady=4, sticky="W")

#Creo l'etichetta e la combobox per il valore "Mq_Lotto"
Label_Lot = tk.Label(window, text = "Metri quadri Lotto: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Lot.grid(row=7, column=1, padx=5, pady=4, sticky="E")
Entry_Lot = ttk.Combobox(window, values = m.get_Lot(), state = 'normal', style="TCombobox", width=25)
Entry_Lot.grid(row=7, column=2, padx=5, pady=4, sticky="W")

#Creo l'etichetta e la combobox per il valore "Mq_Seminterrato"
Label_Basement = tk.Label(window, text = "Metri quadri Seminterrato: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Basement.grid(row=8, column=1, padx=5, pady=4, sticky="E")
Entry_Basement = ttk.Combobox(window, values = m.get_Basement(), state = 'normal', style="TCombobox", width=25)
Entry_Basement.grid(row=8, column=2, padx=5, pady=4, sticky="W")

#Creo l'etichetta e la combobox per il valore "Mq_Calpestabili"
Label_Above = tk.Label(window, text = "Metri quadri Calpestabili: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Above.grid(row=9, column=1, padx=5, pady=4, sticky="E")
Entry_Above = ttk.Combobox(window, values = m.get_Above(), state = 'normal', style="TCombobox", width=25)
Entry_Above.grid(row=9, column=2, padx=5, pady=4, sticky="W")

Entry_Living.current(0)
Entry_Lot.current(0)
Entry_Basement.current(0)
Entry_Above.current(0)

#Creo l'etichetta e la combobox per il valore "AnnoDiCostruzione"
Label_YearC = tk.Label(window, text = "Anno di costruzione: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_YearC.grid(row=10, column=1, padx=5, pady=4, sticky="E")
Entry_YearC = ttk.Combobox(window, values = m.get_Anno_c(), state = 'readonly', style="TCombobox", width=25)
Entry_YearC.grid(row=10, column=2, padx=5, pady=4, sticky="W")
Entry_YearC.current(0)

#Creo l'etichetta e la combobox per il valore "AnnoDiRistrutturazione"
Label_YearR = tk.Label(window, text = "Anno di restauro: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_YearR.grid(row=11, column=1, padx=5, pady=4, sticky="E")
Entry_YearR = ttk.Combobox(window, values = m.get_Anno_r(), state = 'readonly', style="TCombobox", width=25)
Entry_YearR.grid(row=11, column=2, padx=5, pady=4, sticky="W")
Entry_YearR.current(0)

#Creo l'etichetta e lo spinbox per il valore "Piani"
Label_Floor = tk.Label(window, text = "Piani: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Floor.grid(row=12, column=1, padx=5, pady=4, sticky="E")
Entry_Floor = ttk.Spinbox(window,from_ = 0, to = 4, wrap=True, increment=0.5, format='%1.1f', state = 'readonly', style="TSpinbox", width=23)
Entry_Floor.grid(row=12, column=2, padx=5, pady=4, sticky="W")
Entry_Floor.set(1.0)

#Creo l'etichetta e lo spinbox per il valore "Affaccio sul mare"
Label_WF = tk.Label(window, text = "Affaccio sul mare: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_WF.grid(row=13, column=1, padx=5, pady=4, sticky="E")
Entry_WF = ttk.Spinbox(window,from_ = 0, to = 1, wrap=True, increment=1, format='%1d', state = 'readonly', style="TSpinbox", width=23)
Entry_WF.grid(row=13, column=2, padx=5, pady=4, sticky="W")
Entry_WF.set(0)

#Creo l'etichetta e lo spinbox per il valore "Stanze"
Label_Room = tk.Label(window, text = "Stanze: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Room.grid(row=14, column=1, padx=5, pady=4, sticky="E")
Entry_Room = ttk.Spinbox(window,from_ = 1, to = 10, wrap=True, increment=1, format='%1.2f', state = 'readonly', style="TSpinbox", width=23)
Entry_Room.grid(row=14, column=2, padx=5, pady=4, sticky="W")
Entry_Room.set(2.0)

#Creo l'etichetta e lo spinbox per il valore "Vista"
Label_View = tk.Label(window, text = "Vista: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_View.grid(row=15, column=1, padx=5, pady=4, sticky="E")
Entry_View = ttk.Spinbox(window,from_ = 0, to = 5, wrap=True, increment=1, format='%1.1f', state = 'readonly', style="TSpinbox", width=23)
Entry_View.grid(row=15, column=2, padx=5, pady=4, sticky="W")
Entry_View.set(0)

#Creo l'etichetta e lo spinbox per il valore "Condizioni"
Label_Cond = tk.Label(window, text = "Condizione: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Cond.grid(row=16, column=1, padx=5, pady=4, sticky="E")
Entry_Cond = ttk.Spinbox(window,from_ = 1, to = 5, wrap=True, increment=1, format='%1.1f', state = 'readonly', style="TSpinbox", width=23)
Entry_Cond.grid(row=16, column=2, padx=5, pady=4, sticky="W")
Entry_Cond.set(0)

# ComboBox per la predizione
Label_Model = tk.Label(window, text = "Scegli la predizione: ", font=("Segoe UI", 12, "bold"),bg="#0d1b0d", fg="#00ff41")
Label_Model.grid(row=17, column=1, padx=5, pady=4, sticky="E")
ComboBox_Model = ttk.Combobox(window, values = m.get_Modello(), state = 'readonly', style="TCombobox", width=25)
ComboBox_Model.grid(row=17, column=2, padx=5, pady=4, sticky="W")
ComboBox_Model.current(0)

#Creo il pulsante per il passaggio dei valori dell'utente - Design moderno
getValue_button = tk.Button(text="AVVIA PREDIZIONE", 
                            background="#28a745",
                            foreground="white",
                            font=("Segoe UI", 13, "bold"),
                            padx=25,
                            pady=12,
                            relief="flat",
                            cursor="hand2",
                            activebackground="#1e7e34",
                            activeforeground="white",
                            command=predizione_prezzo)
getValue_button.grid(row=18, column=1, columnspan=2, padx=5, pady=15)

# Label per la predizione - Design moderno
Label_Prediction = tk.Label(window, text="", 
                            fg="#00ff41",
                            font=("Segoe UI", 14, "bold"),
                            bg="#0d1b0d",
                            pady=8)
Label_Prediction.grid(row=20, column=1, columnspan=2, padx=5) 

if __name__ == "__main__":
    window.mainloop()
