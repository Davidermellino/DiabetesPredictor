from tkinter import ttk

class HomeView:
    def __init__(self, parent, logo):
        self.parent = parent
        self.logo = logo
        
        self.show_dev_name_btn = ttk.Button(self.parent, text="Show Developers", style="Title.TButton", command=self._show_dev_name)
        self.hide_dev_name_btn = ttk.Button(self.parent, text="Hide Developers", style="Title.TButton", command=self._hide_dev_name)
        self.name_label = ttk.Label(self.parent, text="Developed by: Agus Luca, Ermellino Davide, Paluma Elisa, Pellegrini Alba", style="DevNames.TLabel")

        self._create_widgets()
        
    def _create_widgets(self):
        welcome_label = ttk.Label(self.parent, text="Welcome in DiabetesPredictor", image=self.logo,
        compound="bottom", style="Title.TLabel") 
        welcome_label.pack()
        self.show_dev_name_btn.pack()
        
    def _show_dev_name(self):
        self.show_dev_name_btn.destroy()
        self.hide_dev_name_btn = ttk.Button(self.parent, text="Hide Developers", style="Title.TButton", command=self._hide_dev_name)
        self.hide_dev_name_btn.pack()
        self.name_label = ttk.Label(self.parent, text="Developed by: Agus Luca, Ermellino Davide, Paluma Elisa, Pellegrini Alba", style="DevNames.TLabel")
        self.name_label.pack()
        
    def _hide_dev_name(self):
        self.hide_dev_name_btn.destroy()
        self.name_label.destroy()
        self.show_dev_name_btn = ttk.Button(self.parent, text="Show Developers", style="Title.TButton", command=self._show_dev_name)
        self.show_dev_name_btn.pack()