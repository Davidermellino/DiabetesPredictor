from tkinter import ttk

class HomeView:
    def __init__(self, parent, logo):
        self.parent = parent
        self.logo = logo
        self._create_widgets()
        
    def _create_widgets(self):
        welcome_label = ttk.Label(self.parent, text="Welcome in DiabetesPredictor", image=self.logo,
        compound="bottom", style="Title.TLabel") 
        welcome_label.pack()    
    