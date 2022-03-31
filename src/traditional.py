import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from dataset_generator import loadDataset

if __name__ == "__main__":
    X, Y = loadDataset("variable_length_dataset.csv")

    print(X.shape)
    print(Y.shape)
    reg = LinearRegression()

    # train_set, validation_set = split_dataset(dataset)
    # validation_set, test_set = split_dataset(validation_set, split=0.9)
