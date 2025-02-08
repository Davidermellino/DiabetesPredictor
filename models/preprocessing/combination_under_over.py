
# #importo tecniche di preprocessing
# from models.preprocessing.Sampler import  Sampler
# from models.preprocessing.Scaler import Scaler
# from models.preprocessing.FeatureSelection import FeatureSelection
# from sklearn.model_selection import train_test_split

# # Importo i classificatori
# from classifiers.Artificial_neural_network_sklearn import ArtificialNeuralNetworkSklearn
# from classifiers.Random_forest_custom import RandomForestCustom
# from classifiers.Knn_Custom import KnnCustom
# from classifiers.Decision_tree_sklearn import DecisionTreeSklearn
# from classifiers.Naive_bayes_sklearn import NaiveBayesSklearn

# #from getDataset import getDataset
# from sklearn.metrics import accuracy_score

# def under_50(tecnica, data_x, labels_y):
#     # Suddivisione in training e test set
#     train_x, test_x, train_y, test_y = train_test_split(data_x, labels_y, random_state=0, test_size=0.3)


#     # Inizializzazione delle variabili
#     # processed_x, processed_y = train_x, train_y

#     if tecnica == "RandomUndersampling":
#         processed_x, processed_y = Sampler("rus", train_x, train_y)
    
#     elif tecnica == "NearMiss":
#         processed_x, processed_y = Sampler("nm1", train_x, train_y)

#     elif tecnica == "NearMiss2":
#         processed_x, processed_y = Sampler("nm2", train_x, train_y)

#     return processed_x, processed_y, test_x, test_y

# def over_50(tecnica, train_x, train_y):

#     # Assicurati che PreProcessing sia definito
#     # Inizializzazione delle variabili
#     processed_x, processed_y = train_x, train_y

#     if tecnica == "RandomOverSampling":
#         processed_x, processed_y = Sampler("ros", train_x, train_y)
    
#     elif tecnica == "SMOTE":
#          processed_x, processed_y = Sampler("smote", train_x, train_y)

#     elif tecnica == "ADASYN":
#         processed_x, processed_y  = Sampler("adasyn", train_x, train_y)

#     return processed_x, processed_y


# def modelsAccuracy(model, train_x, train_y, test_x, test_y):
    
#     if model == "DecisionalTree":
#         dt = DecisionTreeSklearn()
#         dt.fit(train_x, train_y)
#         pred_y = dt.predict(test_x)
    
#     elif model == "NaiveBayes":
#         nb = NaiveBayesSklearn()
#         nb.fit(train_x, train_y)
#         pred_y = nb.predict(test_x)

#     elif model == "ANN":
#         aan = ArtificialNeuralNetworkSklearn()
#         aan.fit(train_x, train_y)
#         pred_y = aan.predict(test_x)
            
#     #DA RIVEDERE I CUSTOM
#     elif model == "KNN_Custom":
#         knn_custom = KnnCustom()
#         knn_custom.fit(train_x, train_y)
#         pred_y = knn_custom.predict(test_x)
    
#     elif model == "RandomForest_Custom":
#         random_forest_custom = RandomForestCustom()
#         random_forest_custom.fit(train_x, train_y)
#         pred_y = random_forest_custom.predict(test_x)

#     return accuracy_score(test_y,pred_y)