from tkinter import ttk

#--------CONFIGURAZIONE STILE--------#

def configure_styles():
    style = ttk.Style()
    style.theme_use('clam')
    
    # Nuova palette di colori
    colors = {
        'primary': '#8e44ad',  # Viola scuro
        'secondary': '#c0392b',  # Rosso mattone
        'accent': '#16a085',  # Verde acqua
        'background': '#ecf0f1',  # Grigio chiaro
        'text': '#2c3e50',  # Blu notte
        'hover': '#bdc3c7'  # Grigio intermedio
    }
    
    # Titoli e bottoni principali
    style.configure('Title.TButton', font=('Helvetica', 24, 'bold'), padding=15, 
        background=colors['accent'],  # Verde acqua
        foreground='white'
    )
    style.map('Title.TButton',
        background=[('active', colors['primary'])]  # Rosso mattone al passaggio
    )
    
    style.configure('Title.TLabel',
        font=('Helvetica', 18, 'bold'),
        foreground=colors['primary'],  # Viola scuro
        padding=10
    )
    style.configure('DevNames.TLabel',
        font=('Helvetica', 11, 'bold'),
        foreground=colors['primary'],  # Viola scuro
        padding=10
    )
    
    # Bottoni della sidebar
    style.configure('Sidebar.TButton',
        font=('Helvetica', 12),
        background=colors['background'],  # Grigio chiaro
        foreground=colors['text'],  # Blu notte
        padding=8,
        relief='flat'
    )
    style.map('Sidebar.TButton',
        background=[('active', colors['hover'])],  # Grigio intermedio al passaggio
        foreground=[('active', colors['primary'])]  # Viola scuro
    )
    
    # Area contenuti
    style.configure('Content.TFrame',
        background=colors['background'],
        relief='flat'
    )
    
    style.configure('TCombobox',
        font=('Helvetica', 12),
        padding=5,
        background='white',
        foreground=colors['text'],
        borderwidth=2
    )
    style.map('TCombobox',
        fieldbackground=[('readonly', 'white')],  # Sfondo campo in sola lettura
        background=[('active', colors['hover'])],  # Sfondo quando il menu è aperto
        foreground=[('readonly', colors['text'])]  # Testo della combobox
    )
    
    return style