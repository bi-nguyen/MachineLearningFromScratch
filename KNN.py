import numpy as np


class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, weights = "uniform",p=2):
        self._weights= weights
        self.n_neighbors = n_neighbors
        self._p =p

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self
    def _distance(self,data1,data2): # compute the distance between two samples
        '''
        1: Manhattan distance
        2: Euclidean distance
        '''
        if self._p == 1:
            return np.sum(np.abs(data1-data2))
        elif self._p ==2:
            return np.sqrt(np.sum((data1-data2)**2))
    def _predict_one(self, test):
        result = sorted([(self._distance(x,test),y) for x,y in zip(self.X,self.y)]) # sorted followed by distance
        weights = self._compute_weights(result[:self.n_neighbors])
        d={}
        for distance,y in weights:
            d[y]= d.get(y,0)+distance
        l=[]
        for c,val in d.items():
            l.append((val,c))
        l=sorted(l,reverse=True)
        return l[0][1]
    def _compute_weights(self,distance):
        if self._weights == "uniform":
            return [(1,y) for d,y in distance]
        elif self._weights == "distance":
            matches = [(1,y) for d,y in distance if d==0]
            return matches if matches else [(1/d,y) for d,y in distance ] # ngụ ý nếu có vector 0 có nghĩa là điểm đó nằm trong tập dữ liệu
    def predict(self, X):
        return [self._predict_one(i) for i in X]
    def score(self,X,y):
        Y_predict = self.predict(X)
        return 100*np.mean(Y_predict==y)

def main():
    
    # X_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # y_train = np.array([1,1,1,0,0,0])
    # neighbor = KNeighborsClassifier(n_neighbors=3,weights="distance")
    # neighbor.fit(X_train, y_train)
    # X_test = np.array([[1, 0], [-2, -2],[-1,-1]])
    # knn = KNeighborsClassifier().fit(X_train,y_train)
    # print(knn.predict(X_test))

    # score test
    # X_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # y_train = np.array([1,1,1,0,0,0])
    # neighbor = KNeighborsClassifier().fit(X_train, y_train)
    # X_test = np.array([[1, 0], [-2, -2]])
    # y_test = np.array([0, 0])
    # print(neighbor.score(X_test, y_test))

    # sklearn
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()

    X_train, X_temp, y_train, y_temp = \
        train_test_split(iris.data, iris.target, test_size=.4)
    X_validation, X_test, y_validation, y_test = \
        train_test_split(X_temp, y_temp, test_size=.5)

    neighbor = KNeighborsClassifier().fit(X_train, y_train)

    print(neighbor.score(X_train, y_train))
    print(neighbor.score(X_validation, y_validation))



    return

if __name__ == "__main__":
    main()