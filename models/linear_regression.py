from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def train_simple_linear(df: pd.DataFrame, features: list, target: str):
    X = df[features].values.reshape(-1, 1)
    y = df[target].values  
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_multiple_linear(df: pd.DataFrame, features: list, target: str):
    X = df[features].values
    y = df[target].values
    model = LinearRegression()
    model.fit(X, y)
    return model