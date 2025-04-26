import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def getData(filename):
    """
    Charge et prétraite les données FER-2013.
    """
    Y = []
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

def preprocess_data(data_path='../Data/fer2013.csv'):
    """
    Prétraite les données et les divise en ensembles d'entraînement et de test.
    """
    X, Y = getData(data_path)

    # Redimensionner les images
    N, D = X.shape
    X = X.reshape(N, 48, 48, 1)

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # Encoder les étiquettes en one-hot
    num_classes = len(set(Y))
    y_train = (np.arange(num_classes) == y_train[:, None]).astype(np.float32)
    y_test = (np.arange(num_classes) == y_test[:, None]).astype(np.float32)

    # Sauvegarder les données prétraitées
    np.save('../Data/X_train.npy', X_train)
    np.save('../Data/y_train.npy', y_train)
    np.save('../Data/X_test.npy', X_test)
    np.save('../Data/y_test.npy', y_test)

    print("Données prétraitées et sauvegardées.")
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    preprocess_data()