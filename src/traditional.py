import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model._base import LinearModel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from dataset_generator import loadCsvDataset
from encoding import oneHotEncodeMany, frequencyEncodeMany

if __name__ == "__main__":
    # X, y = loadCsvDataset("variable_length_dataset.csv", encode = oneHotEncodeMany, seqLength = 40)
    X, y = loadCsvDataset("variable_length_dataset.csv", encode = frequencyEncodeMany)
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    models: dict[str, LinearModel] = {
        'ols': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elastic': ElasticNet()
    }

    for name, reg in models.items():
        reg.fit(xTrain, yTrain)

        yTrainPred = reg.predict(xTrain)
        trainRmse = mean_squared_error(yTrain, yTrainPred, squared=False)

        yTestPred = reg.predict(xTest)
        testRmse = mean_squared_error(yTest, yTestPred, squared=False)

        print(name, trainRmse, testRmse)
