# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:28:51 2023

@author: LJMCA
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:52:50 2021

@author: admin
"""


def averaging():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
# importing machine learning models for prediction
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn import svm
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
    model_1 = LinearRegression()
    model_2= svm.SVC(kernel='linear')
    model_3 = RandomForestRegressor( n_estimators=100)
 
    # training all the model on the training dataset
    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)
    model_3.fit(X_train, y_train)
     
    # predicting the output on the validation dataset
    pred_1 = model_1.predict(X_test)
    pred_2 = model_2.predict(X_test)
    pred_3 = model_3.predict(X_test)
    
    pred_final = (pred_1+pred_2+pred_3)/2.0
    print(mean_squared_error(y_test, pred_final))

def bagging(): 
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error 
    # importing machine learning models for prediction
    from sklearn.ensemble import BaggingRegressor
    from sklearn import svm
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
    model = BaggingRegressor(base_estimator=svm.SVC(kernel='linear')) 
    # training model
    model.fit(X_train, y_train)
    # predicting the output on the test dataset
    pred_final = model.predict(X_test)
    # printing the mean squared error between real value and predicted value
    print(mean_squared_error(y_test, pred_final))

def boosting(): 
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error 
    # importing machine learning models for prediction
    from sklearn.ensemble import GradientBoostingRegressor
    data = pd.read_csv("C://My Data//research paper//Darshan//RM//Diabetes_1.csv")
    print(data)
    X = data[data.columns[0:6]]
    print(X)    
    Y = data.Outcome
    print(Y)    
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
    # initializing the boosting module with default parameters
    model = GradientBoostingRegressor()
    # training the model on the train dataset
    model.fit(X_train, y_train)
    # predicting the output on the test dataset
    pred_final = model.predict(X_test)
    # printing the mean squared error between real value and predicted value
    print(mean_squared_error(y_test, pred_final))
print("MENU \n1.Averging \n2.Bagging \n3.Boosting \n4.Exit")
ch = int(input("Enter Your Choice : "))
if ch == 1:
    averaging()
elif ch == 2:
    bagging()
elif ch == 3:
    boosting()
else:
    exit

