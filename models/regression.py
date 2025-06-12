from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

class RegressionModel:
    def __init__(self, model):
        self.model = model
        self.metrics = {}

    def train(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.evaluate()

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        self.metrics = {
            "mae": mean_absolute_error(self.y_test, y_pred),
            "mse": mean_squared_error(self.y_test, y_pred),
            "r2": r2_score(self.y_test, y_pred)
        }

    def print_metrics(self, label=""):
        print(f"{label} | MAE: {self.metrics['mae']:.4f} | MSE: {self.metrics['mse']:.4f} | R2: {self.metrics['r2']:.4f}")

# Usage example for simple/multiple regression:
def train_models(df, features, target, model_type, test_size=0.2, random_state=42):
    models = []
    splits = []
    if model_type == "simple_linear":
        for feature in features:
            model = RegressionModel(LinearRegression())
            X = df[[feature]].values
            y = df[target].values
            model.train(X, y, test_size, random_state)
            model.print_metrics(label=f"Feature: {feature}")
            models.append(model)
            splits.append((model.X_train, model.X_test, model.y_train, model.y_test))


    elif model_type == "multiple_linear":
        model = RegressionModel(LinearRegression())
        X = df[features].values
        y = df[target].values
        model.train(X, y, test_size, random_state)
        model.print_metrics(label="Multiple Linear Regression")
        models = [model]
        splits = [(model.X_train, model.X_test, model.y_train, model.y_test)]

    return models, splits