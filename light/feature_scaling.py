import pandas as pd


class NormalScaler:
    def __init__(self):
        self.max_value = None
        self.min_value = None
        self.column_characteristics: dict = {}

    def fit(self, X: pd.DataFrame):
        self.min_value = X.min(axis=0)
        self.max_value = X.max(axis=0) - self.min_value
        for column in X.columns:
            self.column_characteristics[column] = (X[column].min(), X[column].max())

    def transform(self, X):
        out = pd.DataFrame()
        for column in X.columns:
            out[column] = (X[column] - self.column_characteristics[column][0]) / self.column_characteristics[column][1]
        return out

    def inverse_transform(self, X_scaled):
        out = pd.DataFrame()
        for column in X_scaled.columns:
            out[column] = (X_scaled[column] * self.column_characteristics[column][1] +
                           self.column_characteristics[column][0])
        return out


class StandardScaler:
    def __init__(self):
        self.stdi = None
        self.m = None

    def fit(self, X):
        self.m = X.mean(axis=0)
        self.stdi = X.std(axis=0)

    def transform(self, X):
        X_scaled = (X - self.m) / self.stdi
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_unscaled = (X_scaled * self.stdi) + self.m
        return X_unscaled
