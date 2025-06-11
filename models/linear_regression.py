from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def train_models(df: pd.DataFrame, features: list, target: str, model_type: str):
    """
    Return a list of trained models :
    - one model per feature for the simple linear regression
    - one model for the multiple linear regression
    """
    from sklearn.linear_model import LinearRegression

    models = []
    if model_type == "simple_linear":
        for feature in features:
            X = df[[feature]].values
            y = df[target].values
            model = LinearRegression()
            model.fit(X, y)
            models.append(model)

    elif model_type == "multiple_linear":
        X = df[features].values
        y = df[target].values
        model = LinearRegression()
        model.fit(X, y)
        models = [model]
    return models