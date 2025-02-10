from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


class DecisionTreeSklearn:
    def __init__(self, random_state=42, max_depth=11):
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    
    def fit(self, train_x, train_y):
        #self.tuning_hyperparameters(train_x, train_y)
        self.model.fit(train_x, train_y)
        return self
    
    def predict(self, test_x):
        return self.model.predict(test_x)
    
    def tuning_hyperparameters(self, train_x, train_y):
        
        max_depth_range = list(range(1, 25))
        acc_train = []
        acc_val = []
        
        for depth in max_depth_range:
            clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
            #esegue una 10-fold cross-validation
            scores = cross_validate(estimator=clf, X=train_x, y=train_y, cv=10, n_jobs=4, return_train_score=True, return_estimator=True) 
            score_train = scores['train_score']
            score_val = scores['test_score'] 
            print(f"Depth: {depth}, Train: {score_train.mean()}, Val: {score_val.mean()}")
            acc_train = acc_train + [score_train.mean()]
            acc_val = acc_val + [score_val.mean()]

        best_depth = max_depth_range[acc_val.index(max(acc_val))]
        print("best depth: ", best_depth)

        return best_depth