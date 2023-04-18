import numpy as np 


class LogisticRegression:
    def __init__(self,lr=0.01,epochs=100,lamda=0,batch_size = 10,mode1='LR',mode2='binary'):
        """
        lr : learning rate
        epochs: the number of iteration to find optimal value
        lamda : the penalty factor to prevent overfitting
        batch_size : number of mini batch
        mode1 :
        mode2:
        """
        self._lr=lr
        self._epochs = epochs
        self._lamda = lamda
        self._batchsize = batch_size
        self._mode1 = mode1
        self._mode2 = mode2
        self._Weights = None
        self._bias = None
    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def _softmax(self,z):
        """
        Stable softmax
        
        """
        numerator = np.exp(z-np.max(z,axis=1,keepdims=True))
        denominator = np.sum(numerator,axis=1,keepdims=True)
        return numerator/denominator
    def _logistic_regression(self,X,y,w,b):
        z = np.dot(X,w)+b
        y_predicted = self._sigmoid(z)
        h = y_predicted - y 
        dw = (np.dot(X.T,h) + self._lamda*w)/X.shape[0] + w*self._lamda/X.shape[0]
        db = np.mean(h)
        return dw,db
    def _multi_logistic_regression(self,X,y,w,b):
        """
        X: input
        y: target
        w : weights
        b: bias
        This function is used to compute gradient of multiclassification case
        
        """
        z = np.dot(X,w)+b
        if self._mode2 == 'multi':
            y_predicted = self._sigmoid(z)
        elif self._mode2 == 'softmax':
            y_predicted = self._softmax(z)
        y_predicted[range(X.shape[0]),y]-=1
        dw = (np.dot(X.T,y_predicted) + self._lamda*w)/X.shape[0] + w*self._lamda/X.shape[0]
        db = np.mean(y_predicted,axis=0)
        return dw,db

    def fit(self,X,y):
        n_samples,n_features = X.shape
        if self._mode2 == 'binary':
            self._Weights = np.zeros((n_features,1))
            self._bias = np.zeros((1,1))
        elif (self._mode2 == 'multi') | (self._mode2 =='softmax'):
            self._Weights = np.zeros((n_features,len(np.unique(y))))
            self._bias = np.zeros((1,len(np.unique(y))))
        i=0
        for _ in range(self._epochs):
            if self._mode1 == 'LR':
                if self._mode2 == 'binary':
                    dw,db = self._logistic_regression(X,y,self._Weights,self._bias)
                    self._Weights -= self._lr*dw
                    self._bias -= self._lr*db
                elif (self._mode2 == 'multi') | (self._mode2 =='softmax'):
                    if i==0:
                        print("hello")
                    i+=1
                    dw,db = self._multi_logistic_regression(X,y,self._Weights,self._bias)
                    self._Weights -= self._lr*dw
                    self._bias -= self._lr*db

    def predict(self,X):
        z = np.dot(X,self._Weights)+self._bias
        if self._mode2 == 'softmax':
            y_predicted = self._softmax(z)
        else:
            y_predicted = self._sigmoid(z)
        if self._mode2 == 'binary':       
            y_predicted = np.array([1 if i >=0.5 else 0  for i in y_predicted]).reshape(-1,1)
        elif (self._mode2 == 'multi') | (self._mode2 == 'softmax'):
            y_predicted = np.argmax(y_predicted,axis=1)
        return y_predicted 
    def accuracy(self,y_pred,y):
        return round(100*np.mean(y_pred==y),3)




def main():
    #Imports
    # binary classification
    # from sklearn.model_selection import train_test_split
    # from sklearn import datasets
    # np.random.seed(0)


    # bc = datasets.load_breast_cancer()
    # X, y = bc.data, bc.target

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1234
    # )
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # regressor = LogisticRegression(lr=0.0001, epochs=1000)
    # regressor.fit(X_train, y_train)
    # #print(regressor._weight.shape)
    # predictions = regressor.predict(X_test)

    # print("LR classification accuracy:", regressor.accuracy(y_test, predictions))

    # multi classification
    from scipy.io import loadmat
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml
    from scipy.io import loadmat
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, 
                                                digits.target,
                                               test_size=0.25,
                                               random_state=0)

    regressor = LogisticRegression(lr=0.01, epochs=100,mode2='softmax')
    regressor.fit(X_train, y_train)

    predictions_train = regressor.predict(X_train)
    predictions_test = regressor.predict(X_test)

    print("LR classification accuracy on train set:", regressor.accuracy(y_train, predictions_train))
    print("LR classification accuracy on test set:", regressor.accuracy(y_test, predictions_test))


    return
if __name__ == "__main__":
    main()