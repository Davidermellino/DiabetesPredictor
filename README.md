# DiabetesPredictor

A desktop application for analyzing and predicting diabetes using health data. The project is in its early stages and focuses on building the graphical interface and structuring the code.

---

## **Project Structure**

```bash
│── assets/                          
│   ├── img/                         
│   │   └── logo.png                 
│   └── logo.ico                     
│
│── classifiers/                     
│   └── (future models here)         
│
│── shared/                          
│   ├── config.py                    
│   ├── styles.py                    
│   └── utils.py                     
│
│── views/                           
│   ├── classifiers.py               
│   ├── comparisons.py               
│   ├── dataset_analysis.py          
│   └── home.py                      
│
│── app.py                           
│── main.py                          
│── README.md                        
│── requirements.yml                 
```

---

## **Description of Files and Folders**

### **`assets/`**
Contains static resources such as images and icons used in the graphical interface. For example:
- **`logo.ico`**: Application icon.
- **`img/`**: Folder for additional images (e.g., application logo).

### **`classifiers/`**
Folder reserved for machine learning models. Currently empty, but in the future, it will contain:
- Code for training models.
- Scripts for evaluating and saving models.

### **`shared/`**
Contains shared files used throughout the project:
- **`config.py`**: Configuration constants such as window title, size, and icon path.
- **`styles.py`**: Configures graphical styles for the interface using `ttk.Style`.

### **`views/`**
Each file in this folder represents a screen in the application:
- **`home.py`**: Main page with a welcome message and logo.
- **`dataset_analysis.py`**: Page for dataset analysis (to be implemented).
- **`classifiers.py`**: Page for managing machine learning classifiers (to be implemented).
- **`comparisons.py`**: Page for model comparisons (currently disabled).

### **`app.py`**
Contains the `MainApp` class, which manages the graphical interface and navigation between different screens. It handles:
- Configuring the main window.
- Initializing graphical styles.
- Creating the sidebar with navigation buttons.
- Switching between different screens.

### **`main.py`**
Application entry point. Creates the main window and starts the application.

# Dataset
Download the dataset from [this link](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
) and place it in the root directory of the project.

### **`requirement.yml`**
File specifying the project dependencies. It will be used to create a virtual environment with all the necessary libraries.

---

## **How to Run the Project**

1. Clone the repository:
   ```bash
   git clone https://github.com/Davidermellino/DiabetesPredictor.git

2. Create environment with conda ( optional )

   ```bash
     conda env create -f requirements.yml
3. Run the program
   ```bash
     python ./main.py
