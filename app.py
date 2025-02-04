# IMPORTAZIONE DELLE LIBRERIE ESTERNE
from tkinter import *
from tkinter import ttk # Interfaccie con temi e widget aggiuntivi

#IMPORTAZIONE DELLE VISTE ( Pagine )
from views.home import HomeView
from views.dataset_analysis import DatasetAnalysisView
from views.classifiers import ClassifiersView
from views.comparisons import ComparisonsView

#IMPORTAZIONE DELLE VARIAIBILI GLOBALI ( Solo quelle che servono )
from shared.config import WINDOW_TITLE, WINDOW_SIZE, WINDOW_POSITION, ICON_PATH
from shared.styles import configure_styles

class MainApp:
    def __init__(self, root):
        
        # Configurazione finestra
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE+WINDOW_POSITION)
        self.root.resizable(False, False)
        self.root.iconbitmap(ICON_PATH)
        
        # Configurazione stile
        self.style = configure_styles()

        # Configurazione immagini
        # devo per forza metterla qui, altrimenti viene eliminata dal garbage collector
        self.logo = PhotoImage(file="assets/img/logo.png")

        
        # Frame principali
        self.sidebar = ttk.Frame(root, width=200)
        self.content_area = ttk.Frame(root)
        
        self.sidebar.pack(side=LEFT, fill=Y)
        self.content_area.pack(side=RIGHT, fill=BOTH, expand=True)
        
        # Inizializza componenti
        self._create_sidebar()
        self.show_home()
        
    def _create_sidebar(self):
        
        buttons = {
            "home": self.show_home,
            "dataset analysis": self.show_dataset_analysis_page,
            "classifiers": self.show_classifiers_page,
            "comparisons": self.show_comparisons_page,
        }
        
        for page, command in buttons.items():
            
            button = ttk.Button(self.sidebar, text=page, style="Title.TButton", command=command)
            
            if page == "comparisons":
                print(page) 
                button.state(["disabled"])
                
            button.pack(side=TOP, fill=X, padx=10, pady=10)
        
    def _clear_content(self):
        # Distrugge tutti i widget presenti nell'area di contenuto
        # per poter visualizzare un nuovo widget
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def show_home(self):
        # Configuarazione Immagine
        self._clear_content()
        HomeView(self.content_area, self.logo)

    def show_dataset_analysis_page(self):
        self._clear_content()
        DatasetAnalysisView(self.content_area)
            
    def show_classifiers_page(self):
        self._clear_content()
        ClassifiersView(self.content_area)
        
    
    def show_comparisons_page(self):
        self._clear_content()
        ComparisonsView(self.content_area)
        