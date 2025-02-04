from tkinter import ttk

class DatasetAnalysisView:
    def __init__(self, parent):
        self.parent = parent
        self._create_widgets()
    
    def _create_widgets(self):
        label = ttk.Label(self.parent, text= "You are in dataset_analysis page", style="Title.TLabel")
        label.pack()