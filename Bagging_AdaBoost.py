import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numbers
import random


def load_data(filename):
    data = np.load(filename)
    data_x = data[:, :-1]
    data_y = data[:, -1].astype(int)
    print(f"x: {data_x.shape}, y:{data_y.shape}")
    return data_x, data_y


class Bagging(object):
    def __init__(self, n_classifiers, max_depth):
        '''
        Input:
            n_classifiers: number of trees in the ensemble. int
            max_depth: maximum depth allowed for every tree built. It should not exceed 20. int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        self.n_classifiers = n_classifiers
        self.max_depth = max_depth
    
    def train(self, X, y):
        '''
        Build an ensemble.
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        ###############################
        # TODO: your implementation
        ###############################
        
        row, col = X.shape
        
        self.classiferList = []
        for i in range(self.n_classifiers):
            mask = np.random.choice(row, row)
            X_train, y_train = X[mask], y[mask]
            
            classifier = DecisionTreeClassifier(max_depth=self.max_depth)
            classifier.fit(X_train, y_train)
            
            self.classiferList.append(classifier)
        
    
    def test(self, X):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        
        predList = [classifier.predict(X) for classifier in self.classiferList]
        predArray = np.vstack(predList)
        
        return stats.mode(predArray)[0].reshape(-1)
        
    
    
class Boosting(object):
    
    def __init__(self, n_classifiers, max_depth):
        '''
        Input:
            n_classifiers: the maximum number of trees at which the boosting is terminated. int
            max_depth: maximum depth allowed for every tree built. It should not exceed 2. int
        '''
        if max_depth!=1 and max_depth!=2:
            raise ValueError('max_depth can only be 1 or 2!')
        ###############################
        # TODO: your implementation
        ###############################
        self.n_classifiers = n_classifiers
        self.max_depth = max_depth
        
    def train(self, X, y):
        '''
        Train an adaboost.
        Input:
            X: Feature vector of shape (N, D). N - number of training samples; D - number of features. np ndarray
            y: label vector of shape (N,). np ndarray
        '''
        ###############################
        # TODO: your implementation
        ###############################
        
        row, col = X.shape
        
        self.classifierList = []
        self.alphaList = []
        weight = np.ones(row) / float(row)
        
        for i in range(self.n_classifiers):
            classifier = DecisionTreeClassifier(max_depth=self.max_depth, splitter='random')
            classifier.fit(X, y, sample_weight=weight)
            self.classifierList.append(classifier)
            
            y_pred = classifier.predict(X)
            size = len(y_pred)
            
            mask = 1.0 * (y_pred != y)
            weighted_mask = weight * mask
            error = np.sum(weighted_mask) / np.sum(weight)
            
            if error >= 0.5:
                # reverse the mask
                print('did I enter this condition?')
                mask = 1.0 * (mask == 0.0)
                weighted_mask = weight * mask
                error = np.sum(weighted_mask) / np.sum(weight)
            
            alpha = np.log(np.sqrt((1 - error) / error))
            self.alphaList.append(alpha)
            
            for i in range(size):
                # != is ture == 1.0 == misclassified
                if mask[i] == 1.0:
                    weight[i] *= np.exp(alpha)
                else:
                    weight[i] *= np.exp(-alpha)
            
            # normalize the weight
            if np.sum(weight) != 1.0:
                weight /= np.sum(weight)
                
    
    def test(self, X):
        '''
        Predict labels X. 
        Input:
            X: testing feature vectors of shape (N, D). np array
        Output:
            prediction: label vector of shape (N,). np array, dtype=int
        '''
        ###############################
        # TODO: your implementation
        ###############################
        self.alphaList /= np.sum(self.alphaList)
        pred = np.vstack([alpha * classifier.predict(X) for alpha, classifier in zip(self.alphaList, self.classifierList)])
        
        f = np.sum(pred, axis=0)
        
        # 1 if f(x) >= 0.5
        # 0 if f(x) < 0.5
        return 1.0 * (f >= 0.5)
    
    
# ------------Please do not modify the variable names------------
# set seed and load data
random.seed(165)
np.random.seed(165)
X_train, y_train = load_data('winequality-red-train-2class.npy')
X_test, y_test = load_data('winequality-red-test-2class.npy')

print("Testing accuracy:")
# bagging
bagging_tree = DecisionTreeClassifier(max_depth=15, splitter='best', max_features=1.0)
bagging_tree.fit(X_train, y_train)
random_forest = Bagging(100, 15)
random_forest.train(X_train, y_train)
print("Single Decision Tree: %.5f, Bagging: %.5f" 
      % (np.mean(bagging_tree.predict(X_test)==y_test), np.mean(random_forest.test(X_test)==y_test)))

# boosting
boosting_tree = DecisionTreeClassifier(max_depth=2, splitter='random')
boosting_tree.fit(X_train, y_train)
adaboost = Boosting(1000, 2)
adaboost.train(X_train, y_train)
print('Single Decision Tree: %.5f, AdaBoost: %.5f' 
      % (np.mean(boosting_tree.predict(X_test)==y_test), np.mean(adaboost.test(X_test)==y_test)))
