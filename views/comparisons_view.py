#--------COMPARA CLASSIFICATORI CON DIVERSI PREPROCESSING----------------


#----LIBRERIE ESTERNE
from tkinter import ttk

#----IMPORTAZIONE DELLE VARIAIBILI GLOBALI E FUNZIONI INTERNE
from shared.constants import CLASSIFIERS, PREPROCESSING_NAME
from shared.utils import clear_content
from views.show_model_preProcessed_view import ShowModelPreProcessedView

 
 
class ComparisonsView:
    def __init__(self, parent):
        self.parent = parent
        
        #--------SCELTA CLASSIFICATORE E PREPROCESSING
        self.classifier_choice = None #scelta classificatore con combobox
        self.preprocessing_choice = None #scelta preprocessing con combobox
        
        self._create_widgets()
   
    def _create_widgets(self):
        #titolo
        label = ttk.Label(self.parent, text= "You are in comparisons page", style="Title.TLabel")
        label.pack()
        #frame delle combobox
        frame_choice = ttk.Frame(self.parent)
        frame_choice.pack()
 
        self.classifier_choice = ttk.Combobox(frame_choice, values=CLASSIFIERS , state="readonly")
        self.classifier_choice.pack(side="right")
        self.classifier_choice.current(0)
 
        self.preprocessing_choice = ttk.Combobox(frame_choice, values= PREPROCESSING_NAME, state="readonly")
        self.preprocessing_choice.pack(side="left")
        self.preprocessing_choice.current(0)
       
        #pulsante per trainare il classificatore selezionato con la rispettiva tecnica di preprocessing
        button_train = ttk.Button(self.parent, text="Train Classifier with PreProcessing", style="Title.TButton", command=self.show_model_preprocessed_view)
        button_train.pack(pady=10)
 
    def show_model_preprocessed_view(self):
        #--------MOSTRA LA PAGINA DI PERFORMANCE
        choice_class = self.classifier_choice.get()
        choice_prepro = self.preprocessing_choice.get()
        clear_content(self.parent)
        ShowModelPreProcessedView(self.parent, choice_class, choice_prepro)
 
 