from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN 
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
# from models.preprocessing.combination_under_over import under_50, over_50, modelsAccuracy


#-------SOLUZIONE DI DEEPSEEK-------
# class Sampler:
#     def __init__(self):
#         self.techniques = {
#             "rus": RandomUnderSampler(),
#             "nm1": NearMiss(version=1),
#             "nm2": NearMiss(version=2),
#             "ros": RandomOverSampler(),
#             "smote": SMOTE(),
#             "adasyn": ADASYN()
#         }

#     def apply(self, X, y, technique):
#         if technique not in self.techniques:
#             raise ValueError(f"Tecnica non supportata: {technique}")
#         return self.techniques[technique].fit_resample(X, y)

class Sampler:
    def __init__(self, train_x, train_y):
        
        self.train_x = train_x
        self.train_y = train_y
        
        
    #?-----------UNDERSAMPLING----------------
        
    def randomSampler(self, train_x, train_y):
        rus = RandomUnderSampler()
        train_x_rus, train_y_rus = rus.fit_resample(train_x, train_y)
    
        return train_x_rus, train_y_rus

    #undersampling tramite distanze che preservano i più vicini ai più vicini NearMiss(version=1).
    def nearMissSampler(self, train_x, train_y):
        nm1 = NearMiss(version=1)
        train_x_nm1, train_y_nm1 = nm1.fit_resample(train_x, train_y)
        
        return train_x_nm1, train_y_nm1
    
    #undersampling tramite distanze che preservano i più vicini ai più lontani NearMiss(version=2).
    def nearMissSampler2(self, train_x, train_y):
        nm2 = NearMiss(version=2)
        train_x_nm2, train_y_nm2 = nm2.fit_resample(train_x, train_y)
        
        return train_x_nm2, train_y_nm2
    
    
    #?-----------OVERSAMPLING----------------
    
    #random oversampling tramite RandomOverSampler(...)
    def randomOverSampler(self, train_x, train_y):
        ros = RandomOverSampler()
        train_x_ros, train_y_ros = ros.fit_resample(train_x, train_y)
        
        return train_x_ros, train_y_ros

    
    #oversampling tramite SMOTE(...)
    def smoteSampler(self, train_x, train_y):
        smote = SMOTE()
        train_x_smote, train_y_smote = smote.fit_resample(train_x, train_y)
        
        return train_x_smote, train_y_smote


    #oversampling tramite ADASYN(...) , solo sui dati di training 
    def adasynSampler(self, train_x, train_y):
        adasyn = ADASYN()
        train_x_adasyn, train_y_adasyn = adasyn.fit_resample(train_x, train_y)
        
        return train_x_adasyn, train_y_adasyn
    
    
    #?-----------OVERSAMPLING + UNDERSAMPLING----------------
    
    # def combination(data_x, labels_y, input):
    #     tecniche_undersampling = ["RandomUnderSampling",
    #                                 "NearMiss",
    #                                 "NearMiss2"]
    #     tecniche_oversampling = ["RandomOverSampling",
    #                                 "SMOTE",
    #                                 "ADASYN"]
    
    #     results_models = []

    #     for under in tecniche_undersampling:
    #         for over in tecniche_oversampling:
    #             under_processed_x, under_processed_y, under_test_x, under_test_y = under_50(under, data_x, labels_y)
    #             over_processed_x, over_processed_y, over_test_x, over_test_y = over_50(over, under_processed_x, under_processed_y, under_test_x, under_test_y)
                
    #             accuracy = modelsAccuracy(input, over_processed_x, over_processed_y, over_test_x, over_test_y)
                
    #             if input == "DecisionalTree":
    #                 results_models.append([input,accuracy,under,over, over_test_y.value_counts().values.tolist()])

    #             elif input == "NaiveBayes":
    #                 results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])

    #             elif input == "ANN":
    #                 results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])
                
    #             #DA RIVEDERE I CUSTOM
    #             elif input == "KNN_Custom":
    #                 results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])
                
    #             elif input == "RandomForest_Custom":
    #                 results_models.append([input, accuracy,under,over, over_test_y.value_counts().values.tolist()])

    #     return results_models