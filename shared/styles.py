from tkinter import ttk
#--------CONFIGURAZIONE STILI--------#

def configure_styles():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Helvetica', 24), padding=10)
    style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), foreground='#2c3e50')
    style.configure('Sidebar.TButton', background='#f0f0f0', foreground='#333')
    return style