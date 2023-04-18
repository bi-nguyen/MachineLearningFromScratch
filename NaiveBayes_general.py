import numpy as np

class Gaussian:
    def __init__(self):
        return
    def fit(self,X,y):
        seperate = [[v for v,c in zip(X,y) if c==l] for l in np.unique(y)]
        self.mean = np.array([np.mean(i,axis=0) for i in seperate])
        self.var = np.array([np.var(i,axis=0) for i in seperate])
        self._prior = np.array([np.log(len(i)/X.shape[0]) for i in seperate])
        return self
    def _predict(self,x):
        result = np.exp(-((x-self.mean)**2)/(2*self.var))
        return np.sum(np.log(result/(self.var*np.sqrt(2*np.pi))),axis=1)+self._prior
    def predict(self,X):
        return np.array([np.argmax(self._predict(x)) for x in X])
    def accuracy(self,y_pred,y):
        return 100*np.mean(y_pred==y)


class Gaussian2:
    def fit(self,X,y):
        n_samples,n_features = X.shape
        label = np.unique(y)
        n_class = len(label)
        # creating array for mean,var,prior
        self.mean = np.zeros((n_class,n_features))
        self.var = np.zeros((n_class,n_features))
        self.prior = np.zeros(n_class)
        # calculating mean,var,prior from data
        for idx,c in enumerate(label):
            X_c = X[y==c]
            self.mean[idx,:]=np.mean(X_c,axis=0)
            self.var[idx,:]=np.var(X_c,axis=0)
            self.prior[idx] = np.log(len(X_c)/n_samples)
        return self
    def _predict(self,x):
        result = np.exp(-((x-self.mean)**2)/(2*self.var))
        return np.sum(np.log(result/(self.var*np.sqrt(2*np.pi))),axis=1)+self.prior
    def predict(self,X):
        return np.array([np.argmax(self._predict(x)) for x in X])
    def accuracy(self,y_pred,y):
        return 100*np.mean(y_pred==y)
    
class MultinomialBayes:
    def __init__(self,alpha=1):
        self._alpha = alpha

    def fit(self,X,y):
        self.label=np.unique(y)
        seperate = [[val for val,lab in zip(X,y) if lab ==c] for c in np.unique(y)]
        self.y_prior = np.array([np.log(len(i)/X.shape[0]) for i in seperate])
        N_c = np.array([np.sum(i,axis=0) for i in seperate])
        print(N_c)
        self.lamda = np.log((N_c+self._alpha)/(np.sum(N_c,axis=1,keepdims=True)+self._alpha*N_c.shape[1]))
        return self
    def _predict(self,x):
        result = np.argmax(np.sum(x*self.lamda,axis=1)+self.y_prior)
        return self.label[result]
    def predict(self,X):
        return [self._predict(x) for x in X ]
class MultinomialBayes2(MultinomialBayes):
    def __init__(self,alpha=1):
        super().__init__(alpha)
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.label=np.unique(y)
        # creating the array for 
        self.y_prior = np.zeros(len(self.label))
        self.lamda = np.zeros((len(self.label),n_features))
        for idx,val in enumerate(self.label):
            X_c = X[val==y]
            self.y_prior[idx]=np.log(len(X_c)/n_samples)
            self.lamda[idx,:]=np.log((np.sum(X_c,axis=0)+self._alpha)/(np.sum(X_c)+self._alpha*n_features))
        return self
    def _predict(self,x):
        return self.label[np.argmax(np.sum(x*self.lamda,axis=1)+self.y_prior)]
    def predict(self,X):
        return [self._predict(x) for x in X]


class NaiveBayes:
    def __init__(self,alpha=1):
        self._alpha=alpha
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.label=np.unique(y)
        seperate = [[val for val,l in zip(X,y) if l==c] for c in self.label]
        self.y_prior = np.array([np.log(len(i)/n_samples) for i in seperate])
        N_i = np.array([np.sum(i,axis=0)+self._alpha for i in seperate])
        N_c = np.array([len(i)+self._alpha*n_features for i in seperate]).reshape(-1,1)
        self.lamda = (N_i)/(N_c)
        return self
    def _predict(self,x):
        return self.label[np.argmax(np.sum(x*np.log(self.lamda),axis=1)+np.sum((1-x)*np.log(1-self.lamda),axis=1)+self.y_prior)]
    def predict(self,X):
        return [self._predict(x) for x in X]
class NaiveBayes1:
    def __init__(self,alpha=1):
        self._alpha=alpha
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.label=np.unique(y)
        # creating matrix
        self.y_prior = np.zeros(len(self.label))
        self.lamda = np.zeros((len(self.label,n_features)))
        for idx,val in enumerate(self.label):
            X_c = X[val==y]
            self.y_prior[idx] = np.log(len(X_c)/n_samples)
            self.lamda[idx,:]=np.sum(X_c,axis=1) 
        return self
    def _predict(self,x):
        return self.label[np.argmax(np.sum(x*np.log(self.lamda),axis=1)+np.sum((1-x)*np.log(1-self.lamda),axis=1)+self.y_prior)]
    def predict(self,X):
        return [self._predict(x) for x in X]



    





def main():
    #sport label
    d1= [1,0,0,0,1,1,1,1]
    d2= [0,0,1,0,1,1,0,0]
    d3= [0,1,0,1,0,1,1,0]
    d4= [1,0,0,1,0,1,0,1]
    d5= [1,0,0,0,1,0,1,1]
    d6= [0,0,1,1,0,0,1,1]
    # inf label
    d7= [0,1,1,0,0,0,1,0]
    d8= [1,1,0,1,0,0,1,1]
    d9= [0,1,1,0,0,1,0,0]
    d10=[0,0,0,0,0,0,0,0]
    d11=[0,0,1,0,1,0,1,0]
    #test
    d12=[1,0,0,1,1,1,0,1]
    d13=[0,1,1,0,1,0,1,0]
    X=np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11])
    y = np.array(['S','S', 'S', 'S','S','S','I','I','I','I','I'])
    X_test = np.array([d12,d13])
    model = NaiveBayes().fit(X,y)
    print(model.y_prior)
    print(model.lamda)
    print(model.predict(X_test))
    print("-"*40)
    model1 = NaiveBayes().fit(X,y)
    print(model1.y_prior)
    print(model1.lamda)
    print(model1.predict(X_test))
    print("-"*40)

























    # d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
    # d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
    # d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
    # d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
    # d5 = [2, 0, 0, 1, 0, 0, 0, 1, 0]
    # d6 = [0, 1, 0, 0, 0, 0, 0, 1, 1]

    # X = np.array([d1, d2, d3, d4])
    # X_test = np.array([d5,d6])
    # y = np.array(['B','B', 'B', 'N'])
    # model = MultinomialBayes().fit(X,y)
    # print(model.y_prior)
    # print(model.lamda)
    # print(model.predict(X_test))
    # print("-"*40)
    # model2 = MultinomialBayes2().fit(X,y)
    # print(model2.y_prior)
    # print(model2.lamda)
    # print(model2.predict(X_test))

    


















    # from sklearn.model_selection import train_test_split
    # from sklearn import datasets

    # X, y = datasets.make_classification(
    #     n_samples=1000, n_features=10, n_classes=2, random_state=123
    # )
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=123
    # )
    
    # nb = Gaussian2().fit(X,y)
    # nb1 = Gaussian().fit(X,y)
    # y_pred=nb.predict(X_test)
    # print(nb.accuracy(y_pred,y_test))
    


    # y_pred=nb.predict(X_test)
    # print(nb.accuracy(y_pred,y_test))
    
    return

if __name__ == "__main__":
    main()