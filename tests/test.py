import numpy as np
import pandas as pd
from overrides import override

import light
import light.feature_scaling as fs
from light import PCA
from light.trainer import Trainer


class ClassificationFNN(light.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_classes: int):
        super(ClassificationFNN, self).__init__()
        self.network = light.Sequential(
            light.Linear(input_dim, hidden_dim),
            light.Sigmoid(),
            light.Linear(hidden_dim, hidden_dim),
            light.Tanh(),
            light.Linear(hidden_dim, output_dim),
            light.ReLU(),
            light.Linear(output_dim, num_classes),
            light.Softmax()
        )

    @override
    def forward(self, arg: np.ndarray) -> np.ndarray:
        return self.network(arg)

    @override
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return self.network.backward(d_out)


def main():
    wine_dataset = pd.read_csv("tests/SampleDataset/Wine.csv", header=None)
    wine_dataset.columns = ['type'
        , 'alcohol'
        , 'malicAcid'
        , 'ash'
        , 'ashalcalinity'
        , 'magnesium'
        , 'totalPhenols'
        , 'flavanoids'
        , 'nonFlavanoidPhenols'
        , 'proanthocyanins'
        , 'colorIntensity'
        , 'hue'
        , 'od280_od315'
        , 'proline'
    ]
    wine_dataset = wine_dataset.sample(frac=1).reset_index(drop=True)

    y = wine_dataset.iloc[:, :1]
    X = wine_dataset.drop(['type'], axis=1)

    scaler = fs.NormalScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    pca = PCA(variance_threshold=0.99)
    X_transformed = pd.DataFrame(pca.fit_transform(X_scaled))

    X_train, X_test, y_train, y_test = light.train_test_split(X_transformed, y, 0.8)

    net = ClassificationFNN(
        input_dim=X_train.shape[1],
        hidden_dim=int(X_train.shape[1] * 0.2),
        output_dim=len(y_train["type"].unique()),
        num_classes=len(y_train["type"].unique())
    )

    one_hot_encoder = light.OneHotEncoder(y_train["type"].unique())
    y_train = one_hot_encoder.encode(y_train)
    y_train = pd.DataFrame(y_train)

    optimizer = light.SGD(
        net,
        loss=light.CrossEntropyLoss(),
        learning_rate=0.01,
    )

    trainer = Trainer(optimizer, plot=True)
    net = trainer.train((X_train, y_train), 600)

    # Accuracy on unseen data
    correct = 0
    false = 0
    for idx in range(len(X_test)):
        pred_label = one_hot_encoder.decode(net(X_test.iloc[idx].to_numpy()))
        if pred_label == y_test["type"].iloc[idx]:
            correct += 1
        else:
            false += 1
    print("Accuracy: {:.2f}%".format(correct * 100 / len(X_test)))


if __name__ == '__main__':
    main()
