from tkinter import ttk
from matplotlib import pyplot as plt
import seaborn as sns

from models.Knn_train import KnnTrain
from shared.utils import get_accuracy, get_confusion_matrix, get_other_metrics


class PerformanceView():
    
    def __init__(self, parent, classifier):
        self.parent = parent
        self.classifier = classifier
        self.accuracy = None
        self.confusion_matrix = None
        self.metrics = None
        self.metrics_frame = None
    
        self._create_widgets()
        self.show_metrics()
        
    def _create_widgets(self):
        label = ttk.Label(self.parent, text=f"Performance of {self.classifier}", style="Title.TLabel")
        label.pack()
        
       
        
    def show_metrics(self):
        # --------------CALCOLO METRICHE IN BASE AL CLASSIFICATORE ------------
        if self.classifier == "KNN ( custom )":
            knn = KnnTrain()
            predicted = knn.train_knn()
            self.accuracy = get_accuracy(knn.test_y, predicted)
            self.metrics = get_other_metrics(knn.test_y, predicted)
            self.confusion_matrix = get_confusion_matrix(knn.test_y, predicted)
        
        if self.classifier == "DecisionTree":
            pass
    
        if self.classifier == "RandomForest ( custom )":
            pass
        
        if self.classifier == "Artificial Neural Network":
            pass
        
        if self.classifier == "Naive Bayes":
            pass
        
        if self.metrics_frame:
            self.metrics_frame.destroy()
        
       
        
        # --------------MOSTRO ACCURATEZZA ------------
        accuracy_label = ttk.Label(self.parent, 
                                 text=f"Overall Accuracy: {self.accuracy:.4f}", 
                                 style="Title.TLabel")
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
        
        self.plot_confusion_matrix()
        
    def plot_confusion_matrix(self):
        # Ottieni la matrice di confusione
        cm = self.confusion_matrix

        # Per ogni label, crea un subplot
        num_labels = cm.shape[0]
        fig, axes = plt.subplots(1, num_labels, figsize=(15, 5))

        for i in range(num_labels):
            sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
            axes[i].set_title(f'Label {i}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')

        plt.tight_layout()
        plt.show()