from tkinter import ttk
from views.performance_view import PerformanceView

class ShowModelPreProcessedView: 

    def __init__(self, parent, classifier, preprocessing = None):
        self.parent = parent
        self.classifier = classifier
        self.preprocessing = preprocessing
        self._create_widget()
        
    def _create_widget(self):
        label = ttk.Label(self.parent, text= f"Compare {self.classifier}  without and with {self.preprocessing}", style="Title.TLabel")
        label.pack()

        #----------------------SENZA PREPROCESSING------------------------
        frame_noPre = ttk.Frame(self.parent, relief="ridge")
        frame_noPre.pack(side = "left", fill="both", expand=True)
        label = ttk.Label(frame_noPre, text= f"Before {self.preprocessing}", style="Title.TLabel")
        label.pack()

        pf1 = PerformanceView(frame_noPre, self.classifier, preprocessing=None, show_plot=False)
        button_noPre = ttk.Button(frame_noPre, text="show plot", style="Title.TButton", command=pf1.plot_confusion_matrix)
        button_noPre.pack(padx=10, pady=10)

        #----------------------Con PREPROCESSING------------------------
        frame_PrePro = ttk.Frame(self.parent,relief="ridge")
        frame_PrePro.pack(side=  "right" , fill="both", expand=True)
        label = ttk.Label(frame_PrePro, text= f"After {self.preprocessing}", style="Title.TLabel")
        label.pack()

        pf2 = PerformanceView(frame_PrePro, self.classifier, preprocessing=self.preprocessing, show_plot=False)
        button_PrePro = ttk.Button(frame_PrePro, text="show plot", style="Title.TButton", command=pf2.plot_confusion_matrix)
        button_PrePro.pack(padx=10, pady=10)
        