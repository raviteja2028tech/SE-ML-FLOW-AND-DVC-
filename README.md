# MLflow-DVC-Project

## Overview

This project demonstrates an end-to-end Machine Learning workflow using **DVC (Data Version Control)** and **MLflow**.

The project uses the Iris dataset and a Random Forest Classifier. DVC is used to track the dataset, while MLflow is used to track the machine learning experiment, parameters, metrics, and trained model.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Git
- DVC
- MLflow

## Project Structure

```text
MLflow-DVC-Project/
│
├── data/
│   ├── iris.csv
│   └── iris.csv.dvc
│
├── create_data.py
├── train.py
├── predict.py
├── README.md
├── requirements.txt
├── .gitignore
└── .dvc/
```

## Workflow

```text
Iris Dataset
     ↓
DVC Dataset Versioning
     ↓
Data Preparation
     ↓
Random Forest Model
     ↓
MLflow Experiment Tracking
     ↓
Parameters + Accuracy + Model
     ↓
Model Prediction
```

## 1. Create the Dataset

The Iris dataset is generated using Scikit-learn.

```bash
python create_data.py
```

This creates:

```text
data/iris.csv
```

## 2. Track the Dataset Using DVC

```bash
dvc add data/iris.csv
```

This creates:

```text
data/iris.csv.dvc
```

The `.dvc` file stores the version information for the dataset.

## 3. Train the Model

The project uses a Random Forest Classifier with:

- Number of estimators: 100
- Maximum depth: 3
- Random state: 42
- Test size: 20%

Run:

```bash
python train.py
```

The training script:

1. Loads the Iris dataset.
2. Separates features and target.
3. Splits the dataset into training and testing data.
4. Reads the DVC dataset version.
5. Trains a Random Forest Classifier.
6. Calculates accuracy.
7. Logs the experiment using MLflow.
8. Saves the trained model as an MLflow artifact.

## 4. MLflow Tracking

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Open:

```text
http://localhost:5000
```

The experiment is named:

```text
MLflow_DVC_Iris_Project
```

MLflow records:

### Parameters

```text
dataset_version
n_estimators
max_depth
```

### Metric

```text
accuracy
```

### Model

```text
random_forest_model
```

## 5. Model Prediction

The trained model can be loaded from the MLflow run and used for prediction.

```bash
python predict.py
```

Example output:

```text
Predictions:
[0 0 0 0 0]
```

## 6. DVC and MLflow Roles

### DVC

DVC is used for:

- Dataset versioning
- Tracking changes to the dataset
- Maintaining reproducibility of the data

### MLflow

MLflow is used for:

- Experiment tracking
- Parameter logging
- Metric logging
- Model logging
- Model loading for prediction

## 7. Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Running the Complete Project

Run the following commands in order:

```bash
python create_data.py
```

```bash
dvc add data/iris.csv
```

```bash
python train.py
```

Start MLflow:

```bash
mlflow ui --port 5000
```

Then, in another terminal:

```bash
python predict.py
```

## Results

The Random Forest model achieved an accuracy of:

```text
1.00
```

The trained model was successfully logged to MLflow and subsequently loaded for prediction.

## Conclusion

This project demonstrates how DVC and MLflow can be combined in a machine learning workflow.

DVC manages the dataset version, while MLflow tracks the experiment parameters, model accuracy, and trained model artifact. This provides a reproducible workflow for training and evaluating machine learning models.
