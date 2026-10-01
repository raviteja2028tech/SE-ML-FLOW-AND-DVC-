import os
os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"

import mlflow
import pandas as pd

run_id = "41e99b2a357f44f39d53b775c3284d56"

model_uri = f"runs:/{run_id}/random_forest_model"

model = mlflow.sklearn.load_model(model_uri)

data = pd.read_csv("data/iris.csv")

X = data.drop("target", axis=1)

predictions = model.predict(X.head(5))

print("Predictions:")
print(predictions)