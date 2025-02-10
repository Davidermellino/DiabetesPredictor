from tkinter import ttk
import matplotlib.pyplot as plt

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
        #--------CREO ISTANZA DELLA CLASSE PER ANALIZZARE IL DATASET
        self.analyzer = DatasetAnalyzer() 
        #--------SCELTA FEATURE
        self.features_choice = None
        #--------CREO E MOSTRO I WIDGET ---
        self._create_widgets()             

    def _create_widgets(self):
        # creazione label iniziale
        label = ttk.Label(self.parent, text="You are in dataset analysis page", style="Title.TLabel")
        label.pack()
        

        #-----BOTTONE PER CALCOLARE LE STATISTICHE DEL DATASET
        compute_stats_btn = ttk.Button(
            self.parent,
            text="Compute statistics",
            command=self.show_statistics_page,  # Ora self.analyzer esiste!
            style="Title.TButton"
        )
        compute_stats_btn.pack()
        
        #-----BOTTONE PER CALCOLARE LA MATRICE DI CORRELAZIONE
        show_correlation_matrix_btn = ttk.Button(self.parent, text="Show correlation matrix", command=self.show_correlation_matrix_page, style="Title.TButton")
        show_correlation_matrix_btn.pack(pady=10)        

        # Combobox per la scelta della feature
        self.features_choice = ttk.Combobox(self.parent, values=FEATURES_NAME, state="readonly")
        self.features_choice.pack(pady=10)
        self.features_choice.current(0) 
        # Bottone per confermare la scelta
        show_feature_stast_btn = ttk.Button(self.parent, text="Feature statistics", command=self.show_feature_statistics_page, style="Title.TButton")
        show_feature_stast_btn.pack(pady=10)
       
        
        
    def show_statistics_page(self):
        
        #----CALCOLO DELLE STATISTICHE DEL DATASET
        
        clear_content(self.parent)
        
        statistics = self.analyzer.compute_statistics()
        missing_values = self.analyzer.compute_missing_values_dataset()
        
        ShowDatasetStatisticsView(self.parent,statistics, missing_values)
        
       
    
    def show_correlation_matrix_page(self):
        #----CALCOLO DELLA MATRICE DI CORRELAZIONE
        
        clear_content(self.parent)
        
        correlation_matrix = self.analyzer.compute_correlation_matrix()
        ShowCorrelationMatrixView(self.parent, correlation_matrix)
            
        
    def show_feature_statistics_page(self):
        #----CALCOLO DELLE STATISTICHE DELLA FEATURE SCELTA
        
        choice = self.features_choice.get()
        clear_content(self.parent)
        
        feature_data = self.analyzer.compute_feature_data(choice)
        feature_statistics = self.analyzer.compute_feature_statistics(choice)
        ShowFeatureStatisticsView(self.parent, choice, feature_statistics, feature_data)
        
        
       
        
    
    
    
        
     