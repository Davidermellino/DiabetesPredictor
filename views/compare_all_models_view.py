from tkinter import ttk
from shared.constants import CLASSIFIERS, PREPROCESSING_NAME
from shared.utils import clear_content
from views.show_models_compare_view  import ShowCompareAllModelsView

class CompareAllModelsView:
    def __init__(self, parent):
        self.parent = parent
        self.preprocessing_choice = None
        self._create_widgets()
    
    def _create_widgets(self):
        label = ttk.Label(self.parent, text= "You are in comparisons models page", style="Title.TLabel")
        label.pack()

        frame_choice = ttk.Frame(self.parent)
        frame_choice.pack()

       
        self.preprocessing_choice = ttk.Combobox(frame_choice, values= ["None"]+PREPROCESSING_NAME)
        self.preprocessing_choice.pack(side="left")
        self.preprocessing_choice.current(0)
        

        button_train = ttk.Button(self.parent, text="Train Classifier with PreProcessing", style="Title.TButton", command=self.show_models_compare_view)
        button_train.pack(pady=10)
    

    def show_models_compare_view(self):
        choice_prepro = self.preprocessing_choice.get()
        if choice_prepro == "None": choice_prepro = None
        clear_content(self.parent)
        ShowCompareAllModelsView(self.parent, choice_prepro)
