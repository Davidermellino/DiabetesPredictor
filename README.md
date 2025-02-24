# DiabetesPredictor

A desktop application for analyzing and predicting diabetes using health data. Features a modular architecture with machine learning models, preprocessing pipelines, and GUI views.

---

## **Project Structure**

```bash
│── assets/                          
│   ├── img/                         
│   │   └── logo.png                 
│   └── logo.ico
│
│── models/
│   ├──classifiers/
│      ├── Artificial_neural_network_sklearn.py
│      ├── Decision_tree_sklearn.py
│      ├── Knn_Custom.py
│      ├── Naive_bayes_sklearn.py
│      └── Random_forest_custom.py
│
│   ├──preprocessing/
│      ├── BestCombUnderOver.py
│      ├── BestPreProcessingCombination.py
│      ├── FeatureSelection.py
│      ├── Sampler.py
│      └── Scaler.py
│
│── shared/ 
│   ├── config.py
│   ├── constants.py
│   ├── styles.py
│   └── utils.py
│
│── views/  
│   ├── classifiers_view.py
│   ├── compare_all_models_view.py
│   ├── comparisons_view.py
│   ├── dataset_analysis_view.py
│   ├── home_view.py
│   ├── performance_view.py
│   ├── show_corr_matrix_view.py
│   ├── show_dataset_statistics_view.py
│   ├── show_feature_statistics_view.py
│   ├── show_model_preProcessed_view.py
│   └── show_models_compare_view.py
│
│── app.py
│── main.py
│── requirements.yml
```

## **Key Components**  

### 🔨 **`models/`**  
- Machine learning implementations:  
  - Scikit-learn integration ( Decision Tree, ANN, Naive Bayes ) 
  - Custom algorithms (KNN, Random Forest)  

### ⚙️ **`preprocessing/`**  
- Data pipeline tools:  
  - `FeatureSelection.py`: ANOVA, PCA  
  - `Sampler.py`: SMOTE, RandomUnderSampler  
  - `BestCombUnderOver.py`: Best Sampling Combination ( over and under ) 

### 🧩 **`shared/`**  
- Core utilities:  
  - `constants.py`: Paths and configurations  
  - `styles.py`: Tkinter GUI styling  
  - `utils.py`: Data loading helpers  

### 🖥️ **`views/`**  
- Interactive GUI screens:  
  - `show_corr_matrix_view.py`: Heatmap visualizations  
  - `performance_view.py`: Model metrics dashboard  
  - `show_feature_statistics_view.py`: Distribution plots
  - other views..

---

## **Dataset Preparation**  
1. Download from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  
2. Save to project directory as:  
   `diabetes_012_health_indicators_BRFSS2015`  

---

## **🚀 Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/Davidermellino/DiabetesPredictor.git

2. Go to project directory:
    ```bash
    cd path/to/project/directory  

3. Create environment with conda ( optional ) or install all required package present on requirements.yml
   ```bash
     conda env create -f requirements.yml
4. Run the program
   ```bash
     python ./main.py
