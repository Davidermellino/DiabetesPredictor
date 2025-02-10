from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RandomForestCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=17, random_state=42):
        self.name = "RandomForest"
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.forest = []
        
    def fit(self, train_x, train_y):

        for i in range(self.n_estimators):
            print(f"Training tree {i + 1} of {self.n_estimators}...")
            dtree = DecisionTreeClassifier(random_state=self.random_state)
            
            columns = train_x.columns.to_numpy()
            num_records = train_x.shape[0]
            
            feature_sample = np.random.choice(columns, 6, replace=False)
            sample_indices = train_x.sample(n=num_records, replace=False).index
            
            X = train_x.loc[sample_indices]
            y = train_y.loc[sample_indices]            
            X = X.loc[:, feature_sample]
            
            tree_train_x, _ , tree_train_y, _  = train_test_split(X, y, random_state=0, test_size=0.25)
            
            tree = dtree.fit(tree_train_x, tree_train_y)
            self.forest.append((tree, feature_sample))
            print(f"Tree {i + 1} trained.")
        return self
    
    
        
                
    def predict(self, test_x):
        pred_y = []
        
        for index, record in test_x.iterrows():
            print(f"Predicting record {index + 1} of {test_x.shape[0]}...")
            votes = []
            
            for tree, features in self.forest:
                record_subset = pd.DataFrame([record[features]], columns=features)
                prediction = tree.predict(record_subset)[0]
                votes.append(prediction)
            
            majority_vote = np.bincount(votes).argmax()
            pred_y.append(majority_vote)
        
        return np.array(pred_y)
  

   
    def tuning_hyperparameters(self, train_x, train_y):
        
        max_tree_range = list(range(1, 21, 2))
        results = {
            'n_estimators': [],
            'train_score': [],
            'val_score': []
        }
        
        # Perform cross-validation for different numbers of trees
        for n_trees in max_tree_range:
            print(f"Training with {n_trees} trees...")
            clf = RandomForestCustom(n_estimators=n_trees, random_state=self.random_state)
            scores = cross_validate(
                estimator=clf,
                X=train_x,
                y=train_y,
                cv=5,
                n_jobs=-1,
                return_train_score=True
            )
            
            # Store results
            results['n_estimators'].append(n_trees)
            results['train_score'].append(scores['train_score'].mean())
            results['val_score'].append(scores['test_score'].mean())
            
            print(f"Trees: {n_trees:2d} | Train Score: {scores['train_score'].mean():.4f} | "
                f"Val Score: {scores['test_score'].mean():.4f}")
        
        # Find best number of estimators based on validation score
        best_idx = np.argmax(results['val_score'])
        best_n_estimators = results['n_estimators'][best_idx]
        best_score = results['val_score'][best_idx]
        
        print(f"\nBest performance with {best_n_estimators} trees: {best_score:.4f}")
        
        return best_n_estimators