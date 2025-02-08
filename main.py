from tkinter import Tk
from app import MainApp

# ------------ WIDGET --------------
# Tutte le cose che si vedono a schermo sono widget
# struttura gerarchica di widget
# Esempi: Pulsanti, caselle di testo, etichette, ecc

# ------------ LAYOUT --------------
# Organizzazione dei widget all'interno della finestra
# Esempi: 
#   Grid: Posiziona i widget in una griglia
#   Pack: Posiziona i widget in una direzione, ( orizzontale o verticale ) 
#   Place: Posiziona i widget in una posizione specifica ( come mvw in ncurses )

# ------------ EVENTI --------------
# Esempi:
#   Click di un pulsante
#   Movimento del mouse
#   Pressione di un tasto
#   Resize della finestra
#   Chiusura

# ------------ PIPELINE PER WIDGET  --------------
# 1. Creazione Widget ( con costruttore )
# 2. Configurazione Widget ( con metodi )
# 3. Mostrare Widget ( con pack, grid, place )


if __name__ == "__main__":
    root = Tk()
    app = MainApp(root)
    root.mainloop()

#TODO:
#TUNING
#miglior combinazione
#