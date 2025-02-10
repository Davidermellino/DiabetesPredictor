import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def clear_content(frame):
    # Distrugge tutti i widget presenti nell'area di contenuto
    # per poter visualizzare un nuovo widget
    for widget in frame.winfo_children():
        widget.destroy()
        
def load_dataset():
    return   pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv", encoding="UTF-8")
            
def recast_features(df):
        
    # Recast dei valori delle colonne in modo da renderle più comprensibili
    #UTILIZZATE SOLO PER LA PARTE DI DATA ANALYSIS
    df["GenHlth"] = df["GenHlth"].astype("object")
    df.loc[df.GenHlth == 5, "GenHlth"] = "poor"
    df.loc[df.GenHlth == 4, "GenHlth"] = "fair"
    df.loc[df.GenHlth == 3, "GenHlth"] = "good"
    df.loc[df.GenHlth == 2, "GenHlth"] = "veryGood"
    df.loc[df.GenHlth == 1, "GenHlth"] = "excellent"

    df["Sex"] = df["Sex"].astype("object")
    df.loc[df.Sex == 0, "Sex"] = "F"
    df.loc[df.Sex == 1, "Sex"] = "M"

    int_columns = [
        "Diabetes_012", "HighBP", "HighChol", "CholCheck", "Smoker", 
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", 
        "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
    ]
    df[int_columns] = df[int_columns].astype(int)

    return df


def get_accuracy(test_y, predicted):
        return accuracy_score(test_y, predicted)
    
def get_other_metrics(test_y, predicted):
    # Ottiene la matrice di confusione
    cm = get_confusion_matrix(test_y, predicted)
    total_samples = cm.sum()
    num_classes = cm.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        # Calcola TP, FP, FN, TN per ogni classe
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - (tp + fp + fn)
        
        # Calcola precision, recall e F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2* (precision  * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Salva le metriche per la classe corrente
        metrics[f'class_{i}'] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        }
    
    return metrics

def get_confusion_matrix(test_y, predicted):
    return confusion_matrix(test_y, predicted, labels=[0, 1, 2])