from sklearn.tree import DecisionTreeClassifier

class DecisionTreeSklearn:
    def __init__(self, random_state=42):
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier(random_state=random_state)
        
    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self
        
    def predict(self, test_x):
        return self.model.predict(test_x)