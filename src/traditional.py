from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import mean_squared_error
from dataset_generator import loadCsvDataset
from encoding import oneHotEncodeMany, frequencyEncodeMany

if __name__ == "__main__":
    X, y = loadCsvDataset("./1mill_dataset.csv", frequencyEncodeMany, (oneHotEncodeMany, { 'seqLength' :  60 }))
    coolX, coolY = loadCsvDataset("./vlad.csv", frequencyEncodeMany, (oneHotEncodeMany, { 'seqLength' :  60 }))
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    linModels: dict[str, LinearModel] = {
        'ols': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elastic': ElasticNet()
    }

    for name, reg in linModels.items():
        reg.fit(X, y)

        yPred = reg.predict(coolX)
        rmse = mean_squared_error(coolY, yPred, squared=False)
        print(name, rmse)
    
    # for name, reg in linModels.items():
    #     reg.fit(xTrain, yTrain)
    #
    #     yTrainPred = reg.predict(xTrain)
    #     trainRmse = mean_squared_error(yTrain, yTrainPred, squared=False)
    #
    #     yTestPred = reg.predict(xTest)
    #     testRmse = mean_squared_error(yTest, yTestPred, squared=False)
    #
    #     print(name, trainRmse, testRmse)
