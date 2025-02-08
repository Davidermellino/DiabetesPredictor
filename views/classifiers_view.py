from tkinter import ttk

from shared.constants import CLASSIFIERS
from shared.utils import clear_content

from views.performance_view import PerformanceView

class ClassifiersView:
    def __init__(self, parent):
        self.parent = parent
        self.classifier_choice = None

        
        self._create_widgets()

    def _create_widgets(self):
        label = ttk.Label(self.parent, text= "You are in classifiers page", style="Title.TLabel")
        label.pack()
        
        self.classifier_choice = ttk.Combobox(self.parent, values=CLASSIFIERS)
        self.classifier_choice.pack(pady=10)
        self.classifier_choice.current(0)
        
        button_train = ttk.Button(self.parent, text="Train the selected classifier", style="Title.TButton", command=self.show_performance_page)
        button_train.pack(pady=10)
        
    def show_performance_page(self):
        choice = self.classifier_choice.get()
        clear_content(self.parent)

        PerformanceView(self.parent, choice, preprocessing=None, show_plot=True)
            
