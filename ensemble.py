import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    alpha = None
    n_weakers_limit = None
    weak_classifier = None
    weak_classifier_list = None
    n_weakers = None

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier = weak_classifier

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        self.alpha = np.array([])
        self.n_weakers = 0
        self.weak_classifier_list = []
        n_samples = X.shape[0]
        w = np.ones((n_samples,)) * (1/n_samples)

        y = np.squeeze(y)

        for m in range(self.n_weakers_limit):
            dtclf = self.weak_classifier(max_depth = 1)
            dtclf.fit(X,y,sample_weight = w)

            h = dtclf.predict(X)
            epsilon = np.sum(w[h!=y])
            print('loop at %d, error rate = %f%%' % (m, epsilon*100))
            if epsilon>0.5:
                break
            alpha = np.log((1-epsilon)/epsilon)/2
            tmp = w * np.exp(-alpha*y*h)
            z = np.sum(tmp)
            w = tmp/z
            
            self.alpha = np.append(self.alpha,alpha)
            self.n_weakers+=1
            self.weak_classifier_list.append(dtclf)

            AdaBoostClassifier.save(self,'model_%d.pickle'%m)

        # self.alpha = np.asarray(self.alpha).reshape(-1,1)
        
    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        h = []
        for i in range(self.n_weakers):
            h.append(self.weak_classifier_list[i].predict(X))
        # print("n_weak", self.n_weakers)
        # print("h len",len(h))
        h = np.asarray(h)
        # print(self.alpha.shape)
        # print("h.shape",h.shape)
        # print("shape", (self.alpha * h).shape)
        H = np.sum(self.alpha.reshape(-1,1) * h,axis = 0).reshape(-1,1)
        return H

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        H = self.predict_scores(X)
        H[H>threshold] = 1
        H[H<threshold] = -1
        return H

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
