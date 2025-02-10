from tkinter import ttk, Canvas, Frame
from views.performance_view import PerformanceView
from shared.constants import CLASSIFIERS

class ShowCompareAllModelsView: 
    def __init__(self, parent, preprocessing=None):
        self.parent = parent
        self.preprocessing = preprocessing
        self._create_widget()
        
    def _create_widget(self):
        # CREAZIONE DELLA STRUTTURA
        container = Frame(self.parent)
        container.pack(fill='both', expand=True)
        
        # CREAZIONE DEL CANVAS SCORREVOLE
        canvas = Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # POSIZIONAMENTO DEI WIDGET ( CANVAS E SCROLLBAR )
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # CREAZIONE DEL FRAME SCORREVOLE
        self.scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        
        # AGGIUNTA DEI WIDGET AL FRAME SCORREVOLE
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # ADATTAMENTO DELLA LARGHEZZA DEL FRAME SCORREVOLE ALLA LARGHEZZA DEL CANVAS
        def on_canvas_resize(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        canvas_frame = canvas.create_window((0,0), window=self.scrollable_frame, anchor='nw')
        canvas.bind('<Configure>', on_canvas_resize)

        # AGGIUNTA DEI WIDGET AL FRAME SCORREVOLE
        label = ttk.Label(
            self.scrollable_frame, 
            text=f"Comparison between all models with {self.preprocessing} preprocessing tecnique", 
            style="Title.TLabel"
        )
        label.pack()
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)


        for classifier in CLASSIFIERS:
            pf = PerformanceView(self.scrollable_frame, classifier, preprocessing=self.preprocessing, show_plot=False)
            button = ttk.Button(self.scrollable_frame, text="show plot", style="Title.TButton", command=pf.plot_confusion_matrix)
            button.pack(padx=10, pady=10)