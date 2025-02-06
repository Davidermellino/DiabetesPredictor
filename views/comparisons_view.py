from tkinter import ttk

class ComparisonsView:
    def __init__(self, parent):
        self.parent = parent
        self._create_widgets()
    
    def _create_widgets(self):
        label = ttk.Label(self.parent, text= "You are in comparisons page", style="Title.TLabel")
        label.pack()