from preproc import PreProcessing
from sklearn.model_selection import train_test_split
from classifiers.dtree import decisionTree
from classifiers.naive_bayes import naive_bayes
from classifiers.ann import ann
#!!!!!!!!!!!!!!!  DA AGGIUNGERE I CUSTOM CLASSIFIERS
#from getDataset import getDataset
from sklearn.metrics import accuracy_score

def under_50(tecnica, data_x, labels_y):
    # Suddivisione in training e test set
    train_x, test_x, train_y, test_y = train_test_split(data_x, labels_y, random_state=0, test_size=0.3)

    # Assicurati che PreProcessing sia definito
    preprocessor = PreProcessing(train_x, test_x, train_y, test_y)

    # Inizializzazione delle variabili
    # processed_x, processed_y = train_x, train_y

    if tecnica == "RandomUndersampling":
        processed_x, processed_y = preprocessor.randomSampler()
    
    elif tecnica == "NearMiss":
         processed_x, processed_y = preprocessor.nearMissSampler()

    elif tecnica == "NearMiss2":
        processed_x, processed_y = preprocessor.nearMissSampler2()

    return processed_x, processed_y, test_x, test_y

def over_50(tecnica, train_x, train_y, test_x, test_y):

    # Assicurati che PreProcessing sia definito
    preprocessor = PreProcessing(train_x, test_x, train_y, test_y)

    # Inizializzazione delle variabili
    processed_x, processed_y = train_x, train_y

    if tecnica == "RandomOverSampling":
        processed_x, processed_y = preprocessor.randomOverSampler()
    
    elif tecnica == "SMOTE":
         processed_x, processed_y = preprocessor.smoteSampler()

    elif tecnica == "ADASYN":
        processed_x, processed_y  = preprocessor.adasynSampler()

    return processed_x, processed_y


def modelsAccuracy(model, train_x, train_y, test_x, test_y):
    
    if model == "DecisionalTree":
        pred_y = decisionTree(train_x, train_y, test_x)
    
    elif model == "NaiveBayes":
        pred_y = naive_bayes(train_x, train_y, test_x)

    elif model == "ANN":
        pred_y = ann(train_x, train_y, test_x)
    
    #DA RIVEDERE I CUSTOM
    elif model == "KNN_Custom":
        knn_custom = KnnCustom()
        pred_y = knn_custom(train_x, train_y, test_x)
    
    elif model == "RandomForest_Custom":
        random_forest_custom = RandomForestCustom()
        pred_y = random_forest_custom(train_x, train_y, test_x)

    return accuracy_score(test_y,pred_y)