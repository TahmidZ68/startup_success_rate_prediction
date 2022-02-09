# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 00:15:42 2021

@author: MCH
"""



import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import datasets
import datetime

#Reading Data
df = pd.read_csv('startup data.csv')
df.info()
print(df)
#%%
#null finding
print(df.isnull().sum())
#%%
#Data cleaning
df= df.drop(['Unnamed: 0','state_code','zip_code','category_code','id','city','Unnamed: 6','name','founded_at','closed_at','first_funding_at','last_funding_at','state_code.1','object_id'], axis = 1)
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(axis=0,value=0)
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(axis=0,value=0)
features = df.columns
print(features)
print(df)
#%%
#Preparing Dataset for applying Algorithm
features = [x for x in features if x!='status']
print(features)

train,test = train_test_split(df, test_size = 0.25)
print(len(df))
print(len(train))
print(len(test))
#%%
#DecisionTree
dt = DecisionTreeClassifier(min_samples_split=100, criterion='entropy')

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)


y_pred = dt.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score

score1 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with decision tree ", score1,"%")


text_representation = tree.export_text(dt)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)
    
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt, 
                   feature_names=features,  
                   class_names='status',
                   filled=True)    
#%%
#Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

dt = RandomForestClassifier(min_samples_split=100, criterion='entropy')

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score2 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with Random forest ", score2,"%")

#%%
# KNeighbours Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

dt = KNeighborsClassifier(n_neighbors=100)

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score3 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with KNeighbors ", score3,"%")


#%%
# MLP Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

dt = MLPClassifier(alpha=1e-6,hidden_layer_sizes=(50, 25), random_state=1)

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score4 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with MLP Classifer ", score4,"%")

fig, axes = plt.subplots(5,5)
vmin, vmax = dt.coefs_[0].min(), dt.coefs_[0].max()
for coef, ax in zip(dt.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(200, 100), cmap=plt.cm.gray, vmin=1 * vmin,vmax=1 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())


#%%
#Naive Byes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dt = GaussianNB()

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score5 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with Naive Byes ", score5,"%")





#%%
#SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dt = SVC()

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score6 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with Support Vector Classifer ", score6,"%")



#%%
#Logistics Regression
from sklearn.linear_model import LogisticRegression

dt = LogisticRegression(random_state=10)

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score7 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with Logistic regression ", score7,"%")

#%%
from sklearn.linear_model import SGDClassifier


dt = SGDClassifier(max_iter=100, tol=1e-9)

x_train = train[features]
y_train = train['status']

x_test = test[features]
y_test = test['status']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

score8 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with SGD ", score8,"%")
#%%
data = {'Decision tree':score1, 'Random Forest':score2, 'KNN':score3,'MLP':score4, 
    'Naive Byes':score5,'SVC':score6,'Logistic regression':score7 , 'SGD':score8}
name = list(data.keys())
score = list(data.values())
  
fig = plt.figure(figsize = (12, 6))
 
plt.bar(name, score, color ='orange',
        width = 0.4)
 
plt.xlabel("Used Algorithm")
plt.ylabel("Score of different algorithms")
plt.title("Bar chart on different scores of different algorithms")
plt.show()

