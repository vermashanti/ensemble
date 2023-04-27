# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:52:50 2021

@author: admin
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics

def decision_tree():
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
    clf2 = DecisionTreeClassifier(min_samples_split=2)
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))
    print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted')) 
    fig = plt.figure(figsize=(25,20))
    _= tree.plot_tree(clf2,filled=True)
    fig = plt.figure(figsize=(25,20))
    plt.figure(figsize=(3,3))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 13)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"], size = 10)
    plt.yticks(tick_marks, ["0", "1"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 13)
    plt.xlabel('Predicted label', size = 13)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),  horizontalalignment='center',verticalalignment='center')

def naive_bays():
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
    
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)    
    y_pred = gnb.predict(X_test)
    #print(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))
    print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted')) 
def KNN():
    print("knn")
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)    
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)    
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=100,test_size=0.10)
    print(X_train)
    print(y_train)    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)            
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))
    print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted')) 
  
def svm():
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn import metrics
    import pandas as pd
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)    
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)   
    #splitting X and y into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state =0)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(y_pred)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    plt.figure(figsize=(3,3))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 13)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"], size = 10)
    plt.yticks(tick_marks, ["0", "1"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 13)
    plt.xlabel('Predicted label', size = 13)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),  horizontalalignment='center',verticalalignment='center')
def rf():
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn import metrics
    import pandas as pd
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")  
    X = data[data.columns[0:6]]    
    Y = data.Outcome  
    #splitting X and y into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state =0)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 1000)  
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(3,3))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size = 13)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"], size = 10)
    plt.yticks(tick_marks, ["0", "1"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 13)
    plt.xlabel('Predicted label', size = 13)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),  horizontalalignment='center',verticalalignment='center')
print("MENU \n1.Dicision Tree \n2.Naive Bays \n3.KNN \n4.Linear Regression  \n5.SVM \n6.Random Forest \n7.Exit")
ch = int(input("Enter Your Choice : "))
if ch == 1:
    decision_tree()
elif ch == 2:
    naive_bays()
elif ch == 3:
    KNN()
elif ch == 4:
    LRM()
elif ch == 5:
    svm()
elif ch == 6:
    rf()
else:
    exit

