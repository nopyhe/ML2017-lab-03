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
import matplotlib.pyplot as plt

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

        with open('feature.pickle', "wb") as f:
            pickle.dump((X_list[0], y_list[0], X_list[1], y_list[1]), f)

def preprocessData(data_size = 1000, test_size = 0.25):
    if os.path.exists('train_val_data.pickle'):
        print('Fount train_val_data')
        with open('train_val_data.pickle','rb') as f:
            X_train, X_val, y_train, y_val = pickle.load(f)
    else:
        print('trian_val_data not found. Generating...')

        with open('feature.pickle', 'rb') as f:
            X_nonface, y_nonface, X_face, y_face = pickle.load(f)
        

        idx_nonface = np.random.choice(X_nonface.shape[0],data_size//2,replace=False)
        X_nonface = X_nonface[idx_nonface,:]
        y_nonface = y_nonface[idx_nonface,:]
        X_train_nonface, X_val_nonface, y_train_nonface, y_val_nonface = train_test_split(X_nonface,y_nonface,test_size = test_size)

        idx_face = np.random.choice(X_face.shape[0], data_size//2, replace=False)
        X_face = X_face[idx_face,:]
        y_face = y_face[idx_face,:]
        X_train_face, X_val_face, y_train_face, y_val_face = train_test_split(X_face,y_face,test_size = test_size)
        
        
        X_train = np.concatenate([X_train_nonface,X_train_face])
        y_train = np.concatenate([y_train_nonface,y_train_face])
        idx_train = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx_train,:]
        y_train = y_train[idx_train,:]

        X_val = np.concatenate([X_val_nonface,X_val_face])
        y_val = np.concatenate([y_val_nonface,y_val_face])
        idx_val = np.random.permutation(X_val.shape[0])
        X_val = X_val[idx_val,:]
        y_val = y_val[idx_val,:]

        with open('train_val_data.pickle','wb') as f:
            pickle.dump((X_train, X_val, y_train, y_val),f)

    return X_train, X_val, y_train, y_val


def get_score_of_models(X_train,X_val,y_train,y_val,n_weakers_limit = 25):
    scores = {
        'train':{
            'accuracy':[],
            'precision':[],
            'recall':[],
            'f1':[]
        },
        'val':{
            'accuracy':[],
            'precision':[],
            'recall':[],
            'f1':[]
        }}
    for i in range(n_weakers_limit):
        ada = AdaBoostClassifier.load('model_%d.pickle' % i)
        y_predict_train = ada.predict(X_train)
        y_predict = ada.predict(X_val)

        scores['train']['accuracy'].append(accuracy_score(y_train,y_predict_train))
        scores['train']['precision'].append(precision_score(y_train,y_predict_train))
        scores['train']['recall'].append(recall_score(y_train,y_predict_train))
        scores['train']['f1'].append(f1_score(y_train,y_predict_train))

        scores['val']['accuracy'].append(accuracy_score(y_val,y_predict))
        scores['val']['precision'].append(precision_score(y_val,y_predict))
        scores['val']['recall'].append(recall_score(y_val,y_predict))
        scores['val']['f1'].append(f1_score(y_val,y_predict))

        print(i, accuracy_score(y_train,y_predict_train), accuracy_score(y_val,y_predict), '\t',
        precision_score(y_train,y_predict_train), precision_score(y_val,y_predict),'\t',
        recall_score(y_train,y_predict_train), recall_score(y_val,y_predict),'\t',
        f1_score(y_train,y_predict_train), f1_score(y_val,y_predict))
    
    with open('scores.pickle', "wb") as f:
        pickle.dump(scores, f)

def plot_scores():
    with open('scores.pickle', 'rb') as f:
        scores = pickle.load(f)
    
    plt.plot(scores['train']['accuracy'],marker = 'x',label = 'training set')
    plt.plot(scores['val']['accuracy'],marker = '.',label = 'validation set')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('AdaBoost Performance on Training Set and Validation Set')
    plt.show()
    # plt.savefig('report/adaboost_accuracy.eps',format='eps',dpi=1000)
    

if __name__ == "__main__":
    prepareFeature()
    X_train, X_val, y_train, y_val = preprocessData()

    adaBoost = AdaBoostClassifier(weak_classifier = DecisionTreeClassifier, n_weakers_limit = 25)
    adaBoost.fit(X_train,y_train,save_model=True)

    # get_score_of_models(X_train, X_val, y_train, y_val)
    # plot_scores()

    adaBoost = AdaBoostClassifier.load('model_14.pickle')
    y_predict = adaBoost.predict(X_val)
    
    # print("acc", np.sum(y_predict==y_val)/y_predict.shape[0])
    print(classification_report(y_val,y_predict, target_names = ('nonface', 'face')))
