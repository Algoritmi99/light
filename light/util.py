import math
import random

import numpy as np
import pandas as pd

def z_score(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in df.columns:
        column = df[col]
        mean = column.mean()
        std = column.std()
        out[col] = (column - mean) / std
    return out

def train_test_split(df1: pd.DataFrame, df2: pd.DataFrame, split: float) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    if df1.shape[0] != df2.shape[0]:
        raise ValueError('Number rows in both dataframes must match')

    rowsNum = df1.shape[0]
    chosenIndexes = []
    for i in range(math.floor(rowsNum * split)):
        randomIndex = random.randint(0, rowsNum - 1)
        while randomIndex in chosenIndexes:
            randomIndex = random.randint(0, rowsNum - 1)
        chosenIndexes.append(randomIndex)

    otherIndexes = [i for i in range(len(df1)) if i not in chosenIndexes]

    df1_train = df1.iloc[chosenIndexes]
    df2_train = df2.iloc[chosenIndexes]
    df1_test = df1.iloc[otherIndexes]
    df2_test = df2.iloc[otherIndexes]

    return df1_train, df1_test, df2_train, df2_test


def make_confusion_matrix(y_true: pd.DataFrame, y_pred: list | pd.DataFrame | np.ndarray):
    if len(y_true) != len(y_pred):
        raise ValueError('The length of the two given vectors must match!')

    classes = []
    confusion_map = {}
    for i in range(len(y_pred)):
        if y_true.iloc[i]["type"] not in classes:
            classes.append(y_true.iloc[i]["type"])

    for i in classes:
        confusion_map[i] = {}
        for j in classes:
            confusion_map[i][j] = 0

    for i in range(len(y_pred)):
        confusion_map[y_true.iloc[i]["type"]][y_pred[i]] += 1

    return confusion_map


class OneHotEncoder:
    def __init__(self, class_list):
        self.classes = class_list
        self.class_to_index = {cls: i for i, cls in enumerate(class_list)}

    def encode(self, labels):
        """Converts a label, a list of labels, or a pandas DataFrame column into one-hot encoded vectors."""
        if isinstance(labels, pd.DataFrame):
            labels = labels.squeeze().tolist()
        elif isinstance(labels, str):
            labels = [labels]

        num_classes = len(self.classes)
        encoded = np.zeros((len(labels), num_classes), dtype=int)

        for i, label in enumerate(labels):
            if label in self.class_to_index:
                encoded[i, self.class_to_index[label]] = 1
            else:
                raise ValueError(f"Label '{label}' not found in class list.")

        return encoded if len(encoded) > 1 else encoded[0]

    def decode(self, one_hot_vectors):
        """Converts one-hot encoded vectors back into labels, supporting arrays, lists, and DataFrames."""
        if isinstance(one_hot_vectors, pd.DataFrame):
            one_hot_vectors = one_hot_vectors.values
        elif isinstance(one_hot_vectors, list):
            one_hot_vectors = np.array(one_hot_vectors)
        elif isinstance(one_hot_vectors, np.ndarray) and one_hot_vectors.ndim == 1:
            one_hot_vectors = one_hot_vectors.reshape(1, -1)

        indices = np.argmax(one_hot_vectors, axis=1)
        decoded_labels = [self.classes[i] for i in indices]

        return decoded_labels if len(decoded_labels) > 1 else decoded_labels[0]
