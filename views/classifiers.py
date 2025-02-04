from tkinter import ttk

class ClassifiersView:
    def __init__(self, parent):
        self.parent = parent
        self._create_widgets()

    def _create_widgets(self):
        label = ttk.Label(self.parent, text= "You are in classifiers page", style="Title.TLabel")
        label.pack()