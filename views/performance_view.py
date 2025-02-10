#----IMPORTO LE LIBRERIE PER LA GUI
from tkinter import ttk
from matplotlib import pyplot as plt


#----IMPORTO LA CLASSE PER IL TRAINING
from models.Train_classifier import TrainClassifier

#----IMPORT I MODELLI
from models.classifiers.Knn_Custom import KnnCustom
from models.classifiers.Decision_tree_sklearn import DecisionTreeSklearn
from models.classifiers.Naive_bayes_sklearn import NaiveBayesSklearn
from models.classifiers.Artificial_neural_network_sklearn import ArtificialNeuralNetworkSklearn
from models.classifiers.Random_forest_custom import RandomForestCustom

#----IMPORT FUNIZONI PER TRAINA
from shared.utils import load_dataset
from shared.utils import get_accuracy, get_confusion_matrix, get_other_metrics
from sklearn.metrics import ConfusionMatrixDisplay

class PerformanceView():
    
    def __init__(self, parent, classifier, preprocessing = None, show_plot=False):
        
        self.parent = parent
        
        #--------SCELTA CLASSIFICATORE E PREPROCESSING
        self.classifier = classifier
        self.preprocessing = preprocessing
        
        #--------VARIABILI PER LE METRICHE
        self.accuracy = None
        self.confusion_matrix = None
        self.metrics = None
        self.metrics_frame = None
        self.show_plot = show_plot

        #PER TUNING
        # self.dataset = load_dataset()
        # self.features = self.dataset.iloc[:, 1:] #Salva una matrix con tutti gli attributi
        # self.label = self.dataset.iloc[:, 0] #Salva un array con le Label

        self._create_widgets()
        self.show_metrics()
        
    def _create_widgets(self):
        label = ttk.Label(self.parent, text=f"Performance of {self.classifier}", style="Title.TLabel")
        label.pack()
        
       
        
    def show_metrics(self):
        
        
        # --------------CALCOLO METRICHE IN BASE AL CLASSIFICATORE ------------
        if self.classifier == "KNN ( custom )":
            best_k = 17
            #------PER TUNING-------
            # under_processer = Sampler(self.features, self.label)
            # self.features, self.label = under_processer.nearMissSampler(self.features, self.label)
            # cls = KnnCustom()
            # best_k = cls.tuning_k_weights(self.features, self.label , weights = True) #metti False se non vuoi usare i pesi 
            
            cls = KnnCustom(k=best_k)

        if self.classifier == "DecisionTree":
            
            best_depth=11
            #------PER TUNING-------
            # cls = DecisionTreeSklearn()
            # best_depth = cls.tuning_hyperparameters(self.features,self.label)
           
            cls = DecisionTreeSklearn(max_depth=best_depth)
    
        if self.classifier == "RF ( custom )":
            best_n_tree = 17
            #------PER TUNING-------
            # under_processer = Sampler(self.features, self.label)
            # self.features, self.label = under_processer.nearMissSampler(self.features, self.label)
            # cls = RandomForestCustom()
            # best_n_tree = cls.tuning_hyperparameters(self.features, self.label)
            
            cls = RandomForestCustom(n_estimators=best_n_tree)
        
        if self.classifier == "ANN":
            best_layers = (32,32)
            #------PER TUNING-------
            # cls = ArtificialNeuralNetworkSklearn()
            # best_layers = cls.tuning_hidden_layers(self.features,self.label)
            cls = ArtificialNeuralNetworkSklearn(hidden_layer_sizes=best_layers)
        
        if self.classifier == "Naive Bayes":
            cls = NaiveBayesSklearn()
        
        if self.metrics_frame:
            self.metrics_frame.destroy()
        
       
        #----TRAINO IL CLASSIFICATORE SCELTO E CALCOLO LE METRICHE
        train_classifier = TrainClassifier(cls, self.preprocessing)
        predicted = train_classifier.train()
        self.accuracy = get_accuracy(train_classifier.test_y, predicted)
        self.metrics = get_other_metrics(train_classifier.test_y, predicted)
        self.confusion_matrix = get_confusion_matrix(train_classifier.test_y, predicted)
        
        # --------------MOSTRO ACCURATEZZA ------------
        accuracy_label = ttk.Label(self.parent, text=f"Overall Accuracy: {self.accuracy:.4f}", style="Title.TLabel")
        accuracy_label.pack(pady=5)
        
        
         # Nuovo frame per le metriche
        self.metrics_frame = ttk.Frame(self.parent)
        self.metrics_frame.pack(pady=10, padx=10, fill='x')
        
        
        # Creo frame per ogni classe
        for class_name, class_metrics in self.metrics.items():
            class_frame = ttk.LabelFrame(self.metrics_frame, text=f"Metrics for {class_name}")
            class_frame.pack(pady=5, padx=5, fill='x')
            
            # 2 Colonne per ogni frame
            left_frame = ttk.Frame(class_frame)
            left_frame.pack(side='left', padx=5)
            right_frame = ttk.Frame(class_frame)
            right_frame.pack(side='left', padx=5)
            
            # Prima colonna: Confusion Matrix metrics
            ttk.Label(left_frame, text=f"True Positives: {class_metrics['true_positives']}").pack(anchor='w')
            ttk.Label(left_frame, text=f"False Positives: {class_metrics['false_positives']}").pack(anchor='w')
            ttk.Label(left_frame, text=f"True Negatives: {class_metrics['true_negatives']}").pack(anchor='w')
            ttk.Label(left_frame, text=f"False Negatives: {class_metrics['false_negatives']}").pack(anchor='w')
            
            # Seconda colonna: Performance metrics
            ttk.Label(right_frame, text=f"Precision: {class_metrics['precision']:.4f}").pack(anchor='w')
            ttk.Label(right_frame, text=f"Recall: {class_metrics['recall']:.4f}").pack(anchor='w')
            ttk.Label(right_frame, text=f"F1-Score: {class_metrics['f1_score']:.4f}").pack(anchor='w')
        
        if self.show_plot: self.plot_confusion_matrix()
        
    def plot_confusion_matrix(self):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix)
        disp.plot()
        plt.gcf().canvas.manager.set_window_title(f"Confusion Matrix {self.classifier}")

        plt.show()