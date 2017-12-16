import os
import random
import pickle
from PIL import Image
import numpy as np
from feature import NPDFeature
import multiprocessing
from sklearn.model_selection import train_test_split
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score

def load_img_and_get_feature(filename):
    img = Image.open(filename).convert('L').resize((24, 24), Image.LANCZOS)
    ft = NPDFeature(np.asarray(img)).extract()
    return ft


def prepareFeature():
    if os.path.exists('feature.pickle'):
        print("Found feature pickle file.")
        return
    else:
        print("Feature pickle file not found. Calculating...")
    
    basedir = os.path.join('datasets', 'original')
    
    with multiprocessing.Pool() as p:
        X_list = []
        y_list = []
        for label in ('nonface', 'face'):
            imagedir = os.path.join(basedir, label)
            imagelist = list(map(lambda x: os.path.join(
                imagedir, x), os.listdir(imagedir)))
            imagefeature = np.asarray(
                list(p.map(load_img_and_get_feature, imagelist)))
            y_t = np.ones((imagefeature.shape[0],1))
            if label=='nonface':
                y_t*=-1
            X_list.append(imagefeature)
            y_list.append(y_t)
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        with open('feature.pickle', "wb") as f:
            pickle.dump((X, y), f)

def preprocessData():
    if os.path.exists('train_val_data.pickle'):
        print('Fount train_val_data')
        with open('train_val_data.pickle','rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    else:
        print('trian_val_data not found. Generating...')

        with open('feature.pickle', 'rb') as f:
            X, y = pickle.load(f)
        
        idx = np.random.choice(X.shape[0], 1000, replace=False)

        X = X[idx,:]
        y = y[idx,:]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
        
        with open('train_val_data.pickle','wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test),f)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepareFeature()
    X_train, X_test, y_train, y_test = preprocessData()

    # adaBoost = AdaBoostClassifier(weak_classifier = DecisionTreeClassifier, n_weakers_limit = 100)
    # adaBoost.fit(X_train,y_train)

    # # AdaBoostClassifier.save(adaBoost, )

    # # adaBoost = AdaBoostClassifier.load('model_1.pickle')
    # y_predict = adaBoost.predict(X_test)
    
    # print("acc", np.sum(y_predict==y_test)/y_predict.shape[0])
    # print(classification_report(y_test,y_predict, target_names = ('nonface', 'face')))

    for i in range(100):
        ada = AdaBoostClassifier.load('model_%d.pickle' % i)
        y_predict_train = ada.predict(X_train)
        y_predict = ada.predict(X_test)
        print(i, accuracy_score(y_train,y_predict_train), accuracy_score(y_test,y_predict), '\t',
        precision_score(y_train,y_predict_train), precision_score(y_test,y_predict),'\t',
        recall_score(y_train,y_predict_train), recall_score(y_test,y_predict),'\t',
        f1_score(y_train,y_predict_train), f1_score(y_test,y_predict)

        )
        # precision_score, recall_score, accuracy_score

        # precision_score, recall_score, accuracy_score, f1_score
    pass
