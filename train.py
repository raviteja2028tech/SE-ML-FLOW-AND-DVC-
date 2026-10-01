import mlflow
import mlflow.sklearn
import re 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("data/iris.csv")

X = data.drop("target", axis=1)
y = data["target"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Get DVC dataset version from the .dvc file
with open("data/iris.csv.dvc", "r") as f:
    dvc_file = f.read()

match = re.search(r"md5:\s*([a-f0-9]+)", dvc_file)

if match:
    dvc_version = match.group(1)
else:
    dvc_version = "unknown"


# Create MLflow experiment
mlflow.set_experiment("MLflow_DVC_Iris_Project")


# Model parameters
n_estimators = 100
max_depth = 3


# Start MLflow run
with mlflow.start_run():

    # Log DVC dataset version
    mlflow.log_param("dataset_version", dvc_version)

    # Log model parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
    model,
    "random_forest_model",
    serialization_format="cloudpickle"
)

    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"DVC Dataset Version: {dvc_version}")