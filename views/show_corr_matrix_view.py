from tkinter import ttk
from matplotlib import pyplot as plt
import seaborn as sns

class ShowCorrelationMatrixView:
    def __init__(self, parent, correlation_matrix):
        self.parent = parent
        self.correlation_matrix = correlation_matrix
        
        self._create_widget()
        self._show_correlation_matrix()
        
    def _create_widget(self):
        label = ttk.Label(self.parent, text=f"Correlation Matrix", style="Title.TLabel")
        label.pack()
        
    def _show_correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f')
        plt.show()
        
        