import numpy as np
from regex import E
 
class LinearRegression:
    def __init__(self,lr=0.01,epochs=100,batchsize=10,mode = 'L2',lamda=0):
        self._lr = lr
        self._epochs = epochs
        self._batchsize = batchsize
        self._mode = mode
        self._lamda = lamda
        self._weights = None
        self._bias = None

    def predict(self,X):
        y_predict = np.dot(X,self._weights)+self._bias
        return y_predict
    def _GradientDescent(self,X,y,W,b):
        y_pred = np.dot(X,W)+b
        dw = (np.dot(X.T,(y_pred-y))+self._lamda*np.sum(W))/X.shape[0]
        db = np.sum(y_pred-y)/X.shape[0]
        return dw,db
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self._weights = np.zeros((n_features,1))
        self._bias = np.zeros((1,1))
        if self._mode == 'L2':
            for epoch in range(self._epochs):
                dw,db = self._GradientDescent(X,y,self._weights,self._bias)
                self._weights  -= self._lr*dw
                self._bias -= self._lr*db
        elif self._mode == 'MiniBatch':
            n_batch = int(n_samples/float(self._batchsize))
            for epoch in range(self._epochs):
                idx = np.random.permutation(n_samples)
                for i in range(n_batch):
                    idx_batch = idx[i*self._batchsize:min(self._batchsize*(i+1),n_samples)]
                    X_batch = X[idx_batch]
                    y_batch = y[idx_batch]
                    dw,db = self._GradientDescent(X_batch,y_batch,self._weights,self._bias)
                    self._weights  -= self._lr*dw
                    self._bias -= self._lr*db
        elif self._mode == "Stochatic":
            for epoch in range(self._epochs):
                idx = np.random.permutation(n_samples)
                for i in idx:
                    Xi = X[[i]]
                    yi = y[[i]]
                    dw,db = self._GradientDescent(Xi,yi,self._weights,self._bias)
                    self._weights  -= self._lr*dw
                    self._bias -= self._lr*db

        else:
            print("check your mode.")
    def Mean_square_error(self,y_pred,y):
        return np.mean((y_pred-y)**2)


def main():
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    np.random.seed(0)
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
      
    regressor = LinearRegression(lr=0.01, epochs=1000,mode="MiniBatch",lamda=0)
    regressor.fit(X_train, y_train)
    # print(X_train.shape)
    print(regressor._weights)
    # predictions = regressor.predict(X_test)

    # mse = regressor.Mean_square_error(y_test, predictions)
    # print("MSE:", mse)



    # y_pred_line = regressor.predict(X)
    # cmap = plt.get_cmap("viridis")
    # fig = plt.figure(figsize=(8, 6))
    # m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    # m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    # plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    # plt.show()
    return




if __name__=="__main__":
    main()
                