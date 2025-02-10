from tkinter import ttk
import matplotlib.pyplot as plt

class ShowFeatureStatisticsView:
    def __init__(self, parent, choice, feature_statistics, feature_data):
        self.parent = parent
        self.choice = choice
        self.feature_statistics = feature_statistics
        self.feature_data = feature_data
        
        self._create_widget()
        
        
    def _create_widget(self):
        label = ttk.Label(self.parent, text=f"Statistics for {self.choice}", style="Title.TLabel")
        label.pack()
        
        # CALCOLO DELLE STATISTICHE   
        for statistic, value in self.feature_statistics.items():     
            label = ttk.Label(self.parent, text=f"{statistic}: {value}", style="Title.TLabel")
            label.pack()

        # PLOTTA L'ISTOGRAMMA NEL CASO DI FEATURE DI TIPO int64 o int32, IL BOXPLOT E L'ISTOGRAMMA ALTRIMENTI
        if self.feature_data.dtype == "float64": 
            self._plot_boxplot_and_histogram(self.feature_data)
        elif self.feature_data.dtype == "int64" or self.feature_data.dtype == "int32" or self.feature_data.dtype == "object":  # Controlla se è di tipo in # Controlla se è di tipo stringa (object)
            self._plot_histogram(self.feature_data)
        
    def _plot_histogram(self, feature_data):
        
        # PLOTTA L'ISTOGRAMMA
        feature_data.value_counts().plot(kind='bar', color=['red', 'blue', 'green','purple', 'yellow'], edgecolor='black')

        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(feature_data.name)

        plt.show()
        
    def _plot_boxplot_and_histogram(self, feature_data):
        
        # PLOTTA IL BOXPLOT E L'ISTOGRAMMA
        plt.subplot(1, 2, 1)
        plt.boxplot(feature_data)
        plt.title(f'Boxplot di {feature_data.name}')
        
        plt.subplot(1, 2, 2)
        plt.hist(feature_data, bins=50, density=True, facecolor='green')
        plt.title(f'Histogram di {feature_data.name}')
        
        plt.show()    