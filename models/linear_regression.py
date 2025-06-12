from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def train_models(df: pd.DataFrame, features: list, target: str, model_type: str, test_size: float = 0.2, random_state: int = 42):
    """
    Return a list of trained models :
    - one model per feature for the simple linear regression
    - one model for the multiple linear regression
    """
    models = []
    splits = []

    if model_type == "simple_linear":
        for feature in features:
            X = df[[feature]].values
            y = df[target].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Feature: {feature} | MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}")
            models.append(model)
            splits.append((X_train, X_test, y_train, y_test))


    elif model_type == "multiple_linear":
        X = df[features].values
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Multiple Linear Regression | Features: {features[0]} | MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}")
        models = [model]
        splits = [(X_train, X_test, y_train, y_test)]

    return models, splits