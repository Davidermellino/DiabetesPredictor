from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns

from models.dataset_analysis import DatasetAnalyzer
from shared.constants import FEATURES_NAME



class DatasetAnalysisView:
    def __init__(self, parent):
        self.parent = parent
        self.analyzer = DatasetAnalyzer()  # Creo prima l'istanza
        self.features_choice = 0
        
        self._create_widgets()             # Poi creo i widget
         

    def _create_widgets(self):
        #creazione label iniziale
        label = ttk.Label(self.parent, text="You are in dataset_analysis page", style="Title.TLabel")
        label.pack()

        #Creazione pulsanti 
        button = ttk.Button(
            self.parent,
            text="Compute statistics",
            command=self.show_statistics,  # Ora self.analyzer esiste!
            style="Title.TButton"
        )
        button.pack()


        # Creazione della Combobox
        self.features_choice = ttk.Combobox(self.parent, values=FEATURES_NAME)
        self.features_choice.pack(pady=10)
        self.features_choice.current(0) 

        # Bottone per confermare la scelta
        btn = ttk.Button(self.parent, text="Plot this feature", command=self.show_plot, style="Title.TButton")
        btn.pack(pady=10)
        
        
    def show_statistics(self):
        
        statistics = self.analyzer.compute_statistics()
        num_record = statistics["records count"]
        
        for statistic, value in statistics.items():
            
            if statistic != "records count" and statistic != "features count":
                label = ttk.Label(self.parent, text=f"{statistic}: {value} ({value/num_record*100:.2f}%)", style="Title.TLabel")
                label.pack()
            else:
                label = ttk.Label(self.parent, text=f"{statistic}: {value}", style="Title.TLabel")
                label.pack()
            
        
    def show_plot(self):
        
        feature_data = self.analyzer.compute_feature_data(self.features_choice.get())

        print(f"Feature data: {feature_data, feature_data.dtype}")

        if feature_data.dtype == "float64":  # Controlla se è di tipo float
            #creo il subplot
            self.plot_boxplot_and_histogram(feature_data)
        elif feature_data.dtype == "int64" or feature_data.dtype == "object":  # Controlla se è di tipo in # Controlla se è di tipo stringa (object)
            self.plot_histogram(feature_data)
        
    def plot_histogram(self, feature_data):
        
        feature_data.value_counts().plot(kind='bar', color=['red', 'blue', 'green','purple', 'yellow'], edgecolor='black')

        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(feature_data.name)

        plt.show()
        
    def plot_boxplot_and_histogram(self, feature_data):
        
        plt.subplot(1, 2, 1)
        plt.boxplot(feature_data, labels=[self.features_choice])
        plt.title(f'Boxplot di {feature_data.name}')
        
        plt.subplot(1, 2, 2)
        plt.hist(feature_data, bins=50, density=True, facecolor='green')
        plt.title(f'Istogramma di {feature_data.name}')
        
        plt.show()    
    
    def plot_correlation_matrix(self):
        corr_matrix = self.analyzer.compute_correlation_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f')
        plt.show()
