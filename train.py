import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Real_Estate_Pricing_Model")

def train_and_log():
    with mlflow.start_run() as run:
        print(f"Starting Training Run ID: {run.info.run_id}")
        
        np.random.seed(42)
        X = np.random.rand(1000, 3) * [5000, 5, 4]
        y = X[:, 0] * 150 + X[:, 1] * 50000 + X[:, 2] * 25000
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        n_estimators = 100
        max_depth = 10
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        
        print("Fitting model...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mlflow.log_metric("rmse", rmse)
        print(f"Validation RMSE: ${rmse:,.2f}")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model successfully trained, logged, and exported to models/model.pkl")

if __name__ == "__main__":
    train_and_log()
