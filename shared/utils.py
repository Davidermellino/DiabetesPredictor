import pandas as pd
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix


def clear_content(frame):
    # Distrugge tutti i widget presenti nell'area di contenuto
    # per poter visualizzare un nuovo widget
    for widget in frame.winfo_children():
        widget.destroy()
        
def load_dataset():
    return   pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv", encoding="UTF-8")
            
def recast_features(df):
        
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
    # Ottiene la matrice di confusione per ogni classe
    conf_matrices = get_confusion_matrix(test_y, predicted)
    # Dizionario per salvare tutte le metriche
    metrics = {}
    
    # Calcola le metriche per ogni classe
    for i, conf_matrix in enumerate(conf_matrices):
        # Estrae i valori dalla matrice di confusione
        tn, fp = conf_matrix[0]
        fn, tp = conf_matrix[1]
        
        # Calcola precision, recall e f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Salva le metriche per questa classe
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
    return multilabel_confusion_matrix(test_y, predicted)

