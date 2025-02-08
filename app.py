# IMPORTAZIONE DELLE LIBRERIE ESTERNE
from tkinter import *
from tkinter import ttk # Interfaccie con temi e widget aggiuntivi

#IMPORTAZIONE DELLE VISTE ( Pagine )
from views.home_view import HomeView
from views.dataset_analysis_view import DatasetAnalysisView
from views.classifiers_view import ClassifiersView
from views.comparisons_view import ComparisonsView
from views.compare_all_models_view import CompareAllModelsView

#IMPORTAZIONE DELLE VARIAIBILI GLOBALI ( Solo quelle che servono )
from shared.config import WINDOW_TITLE, WINDOW_SIZE, WINDOW_POSITION, ICON_PATH
from shared.styles import configure_styles
from shared.utils import clear_content

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
            "compare all models":self.show_compare_all_page,
        }
        
        for page, command in buttons.items(): #crea un bottone per ogni item nel dizionario e gli assegna l'azione command
            
            button = ttk.Button(self.sidebar, text=page, style="Title.TButton", command=command)
                
            button.pack(side=TOP, fill=X, padx=10, pady=10)
        
  

    def show_home(self):
        # Configuarazione Immagine
        clear_content(self.content_area)
        HomeView(self.content_area, self.logo)

    def show_dataset_analysis_page(self):
        clear_content(self.content_area)
        DatasetAnalysisView(self.content_area)
            
    def show_classifiers_page(self):
        clear_content(self.content_area)
        ClassifiersView(self.content_area)
        
    
    def show_comparisons_page(self):
        clear_content(self.content_area)
        ComparisonsView(self.content_area)
    
    
    def show_compare_all_page(self):
        clear_content(self.content_area)
        CompareAllModelsView(self.content_area)
        