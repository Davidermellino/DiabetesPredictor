from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns

from models.dataset_analysis import DatasetAnalyzer
from shared.constants import FEATURES_NAME
from shared.utils import clear_content

from views.show_feature_statistics_view import ShowFeatureStatisticsView
from views.show_corr_matrix_view import ShowCorrelationMatrixView
from views.show_feature_statistics_view import ShowFeatureStatisticsView
from views.show_dataset_statistics_view import ShowDatasetStatisticsView


class DatasetAnalysisView:
    def __init__(self, parent):
        self.parent = parent
        self.analyzer = DatasetAnalyzer()  # Creo prima l'istanza
        self.features_choice = None
        
        self._create_widgets()             # Poi creo i widget
         

    def _create_widgets(self):
        # creazione label iniziale
        label = ttk.Label(self.parent, text="You are in dataset_analysis page", style="Title.TLabel")
        label.pack()
        

        #Creazione pulsanti 
        compute_stats_btn = ttk.Button(
            self.parent,
            text="Compute statistics",
            command=self.show_statistics_page,  # Ora self.analyzer esiste!
            style="Title.TButton"
        )
        compute_stats_btn.pack()
        
        
        show_correlation_matrix_btn = ttk.Button(self.parent, text="Show correlation matrix", command=self.show_correlation_matrix_page, style="Title.TButton")
        show_correlation_matrix_btn.pack(pady=10)        

        self.features_choice = ttk.Combobox(self.parent, values=FEATURES_NAME)
        self.features_choice.pack(pady=10)
        self.features_choice.current(0) 
        # Bottone per confermare la scelta
        show_feature_stast_btn = ttk.Button(self.parent, text="Feature statistics", command=self.show_feature_statistics_page, style="Title.TButton")
        show_feature_stast_btn.pack(pady=10)
       
        
        
    def show_statistics_page(self):
        
        clear_content(self.parent)
        
        statistics = self.analyzer.compute_statistics()
        missing_values = self.analyzer.compute_missing_values_dataset()
        
        ShowDatasetStatisticsView(self.parent,statistics, missing_values)
        
       
    
    def show_correlation_matrix_page(self):
        clear_content(self.parent)
        
        correlation_matrix = self.analyzer.compute_correlation_matrix()
        ShowCorrelationMatrixView(self.parent, correlation_matrix)
            
        
    def show_feature_statistics_page(self):
        
        choice = self.features_choice.get()
        clear_content(self.parent)
        
        feature_data = self.analyzer.compute_feature_data(choice)
        feature_statistics = self.analyzer.compute_feature_statistics(choice)
        ShowFeatureStatisticsView(self.parent, choice, feature_statistics, feature_data)
        
        
       
        
    
    
    
        
     
