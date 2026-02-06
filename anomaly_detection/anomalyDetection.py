import pandas as pd
import numpy as np
import matplotlib as plt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (log_loss, mean_squared_error)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,accuracy_score,f1_score,classification_report,precision_recall_curve,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import os
from sklearn.svm import SVC
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

from numpy import loadtxt
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
import keras

from sklearn.decomposition import PCA
import mca
import prince

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

import datetime
import time


def writeData(dir, data):
    file1 = open(dir,"w")
    for i in range(0, len(data)):
        file1.write(data[i]+'\n\n\n')
    file1.close()

def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end

def group_time(data):
  A = [datetime.time(0, 0, 0), datetime.time(3, 0, 0)]
  B = [datetime.time(3, 0, 0), datetime.time(6, 0, 0)]
  C = [datetime.time(6, 0, 0), datetime.time(9, 0, 0)]
  D = [datetime.time(9, 0, 0), datetime.time(12, 0, 0)]
  E = [datetime.time(12, 0, 0), datetime.time(15, 0, 0)]
  F = [datetime.time(15, 0, 0), datetime.time(18, 0, 0)]
  G = [datetime.time(18, 0, 0), datetime.time(21, 0, 0)]
  H = [datetime.time(21, 0, 0), datetime.time(0, 0, 0)]
  groupTime = []
  for i in range(0, len(data)):
    temp = datetime.datetime.strptime(data[i], '%H:%M').time()
    if (time_in_range(A[0], A[1], temp)):
      groupTime.append('A')
    elif (time_in_range(B[0], B[1], temp)):
      groupTime.append('B')
    elif (time_in_range(C[0], C[1], temp)):
      groupTime.append('C')
    elif (time_in_range(D[0], D[1], temp)):
      groupTime.append('D')
    elif (time_in_range(E[0], E[1], temp)):
      groupTime.append('E')
    elif (time_in_range(F[0], F[1], temp)):
      groupTime.append('F')
    elif (time_in_range(G[0], G[1], temp)):
      groupTime.append('G')
    else:
      groupTime.append('H')

  return groupTime

def preprocessing(filename):
    fraud_dataset = pd.read_csv(filename)
    fraud_1 = fraud_dataset[:100000]
    newPath = 'Data Output'
    if not os.path.exists(newPath):
        os.makedirs(newPath)

    fraud_1[['Date','Time in Hour']] = fraud_1.Time.str.split(' ', expand=True)

    fraud_1['Date'] = pd.to_datetime(fraud_1['Date'])
    fraud_1['Day of Week'] = fraud_1['Date'].dt.day_name()


    fraud_1 = fraud_1.drop(columns=['Time'], axis=1)
    fraud_1 = fraud_1.rename(columns = {'Time in Hour':'Time'})
    fraud_1['Group Time'] = group_time(fraud_1['Time'])

    data = ['Detail Type: \n'+str(fraud_1.type.value_counts()), 'Detail Type Associated with Fraud: \n'+str(fraud_1.groupby(['type','isFraud']).size())]
    dir = 'Data Output/Preprocessing.txt'
    writeData(dir, data)

    dfTransfer = fraud_1.loc[fraud_1.type == 'TRANSFER']
    dfFraudTransfer = fraud_1.loc[(fraud_1.isFraud == 1) & (fraud_1.type == 'TRANSFER')]
    dfFraudCashout = fraud_1.loc[(fraud_1.isFraud == 1) & (fraud_1.type == 'CASH_OUT')]


    fraud_1 = fraud_1.rename(columns = {'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig'})

    fraud_1['errorOrig'] = fraud_1['amount'] + fraud_1['newBalanceOrig'] -fraud_1['oldBalanceOrig']
    fraud_1['errorDest'] = fraud_1['amount'] + fraud_1['oldbalanceDest'] - fraud_1['newbalanceDest']

    filename = 'Data Output/DatasetFraud(After Preprocessing).csv'
    fraud_1.to_csv(filename, index=None)

    fraud_1.drop(['Transaction ID','Account ID','No. KTP','Address', 'Device ID','IP Address','Country ID','Destination ID','Phone Number', 'Time', 'Date', 'isFraud'],axis=1,inplace=True)

    #MCA
    mca = prince.MCA()
    X = fraud_1[['type', 'Day of Week', 'Group Time']]
    mca = mca.fit(X) # same as calling ca.fs_r(1)
    mca.plot_coordinates(X,
                         row_points_alpha=.2,
                         figsize=(10, 10),
                         show_column_labels=True
                        );
    mca = mca.transform(X) # same as calling ca.fs_r_sup(df_new) for *another* test set.
    mca = pd.DataFrame(mca)
    mca = mca.rename(columns = {0:'MCA 1', 1:'MCA 2'})

    fraud_1.drop(['type', 'Day of Week', 'Group Time'],axis=1,inplace=True)

    Y = StandardScaler().fit_transform(fraud_1)

    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(Y)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA 1', 'PCA 2', 'PCA 3', 'PCA 4'])

    final_dataset = pd.concat([principalDf, mca], axis=1, join='inner')

    kmeans = KMeans(2, random_state=1)
    clusters = kmeans.fit_predict(final_dataset)
    labels = pd.DataFrame(clusters)

    labeled_dataset = pd.concat([final_dataset, labels], axis=1, join='inner')
    labeled_dataset = labeled_dataset.rename(columns = {0:'isFraud'})

    y = labeled_dataset['isFraud']
    features = labeled_dataset.drop(['isFraud'],axis=1,inplace=True)
    features = labeled_dataset

    x_train,x_test,y_train,y_test = train_test_split(features,y,test_size=0.2)

    return x_train,x_test,y_train,y_test, fraud_1


def randomForest(x_train,x_test,y_train,y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf = clf.fit(x_train, y_train)
    rf_prediction = clf.predict(x_test)

    data = ["RF Accuracy Score: "+ str(accuracy_score(y_test,rf_prediction)), "RF Classification Report: \n"+str(classification_report(y_test, rf_prediction))]
    dir = 'Data Output/Random Forest Result.txt'

    writeData(dir, data)

    return clf

def SVCModel(x_train,x_test,y_train,y_test):
    svc_rbf = SVC(kernel='rbf', C=1e2, gamma=0.00001)
    svc_rbf.fit(x_train, y_train)
    svm_prediction = svc_rbf.predict(x_test)

    data = ["SVC Accuracy Score: "+ str(accuracy_score(y_test,svm_prediction)), "SVC Classification Report: \n"+str(classification_report(y_test, svm_prediction))]
    dir = 'Data Output/SVC Result.txt'

    writeData(dir, data)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def lrModel(x_train,x_test,y_train,y_test):
    sm = SMOTE(random_state=2)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
    parameters = {
        'C': np.linspace(1, 10, 10)
                 }
    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
    clf.fit(x_train_res, y_train_res.ravel())

    lr1 = LogisticRegression(C=1,penalty='l2', verbose=5)
    lr1.fit(x_train_res, y_train_res.ravel())


    y_train_pre = lr1.predict(x_train)

    cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
    plt.savefig('Data Output/Confusion Matrix of LR(Train).png', dpi=1024)
    plt.close()


    y_pre = lr1.predict(x_test)

    cnf_matrix = confusion_matrix(y_test, y_pre)

    #print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
    plt.savefig('Data Output/Confusion Matrix of LR(Test).png', dpi=1024)
    plt.close()


    tmp = lr1.fit(x_train_res, y_train_res.ravel())

    y_pred_sample_score = tmp.decision_function(x_test)


    fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

    roc_auc = auc(fpr,tpr)

    # Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Data Output/Receiver Operating Characteristic.png', dpi=1024)
    plt.close()

    data = ["CLF best Parameter: \n"+str(clf.best_params_), "\n Recall metric in the train dataset: "+str(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1]))+"%", "\n Recall metric in the testing dataset: "+str((100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))+"%"]
    dir = 'Data Output/LR Result.txt'
    writeData(dir, data)

def dnnModel(x_train,x_test,y_train,y_test):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(4,)))
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    sgd = SGD(lr=0.001, momentum=0.8)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=150, batch_size=10)

    _, accuracyTrain = model.evaluate(x_train, y_train)

    _, accuracyTest = model.evaluate(x_test, y_test)
    predictions = model.predict_classes(x_test)
    predictions = predictions.ravel()
    data = ["Accuracy Train: "+str(accuracyTrain*100), "Accuracy Test: "+str(accuracyTest*100), "\nClassification Report Test: \n"+str(classification_report(y_test, predictions)) ]
    dir = 'Data Output/DNN Result.txt'

    writeData(dir, data)


def main():
    filename = 'Dataset Fraud (New_Final).csv'
    x_train,x_test,y_train,y_test, fraud_1 = preprocessing(filename)
    clf = randomForest(x_train,x_test,y_train,y_test)
    SVCModel(x_train,x_test,y_train,y_test)
    lrModel(x_train,x_test,y_train,y_test)
    dnnModel(x_train,x_test,y_train,y_test)

if __name__ == '__main__':
    main()
