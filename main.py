# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:38:52 2021

@author: Ozan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv("C:/Users/Ozan/Desktop/heart_attack/heart.csv")



df = data.rename(columns={'trtbps':'rest_bps','thalachh':'thal',
                              'exng':'exang','caa':'ca','output':'mark'})



df1 = df.drop(labels=['slp','thall'], axis=1, inplace=False)

columns = df1.columns


print(df1.isnull().sum())




corr_Pearson = df1.corr(method="pearson")

figure = plt.figure(figsize=(12,8))
sns.heatmap(corr_Pearson,vmin=-1,vmax=+1,cmap='Blues',annot=True, 
            linewidths=1,linecolor = 'white')
plt.title('Pearson Correlation')
plt.show()


'''
df1.hist(figsize=(15,10))
plt.show()


j = 0
for i in columns:
    j = j+1
    if(j!=len(columns)):
        sns.lineplot(x='mark', y=i,data=df1)
        plt.show()
    

figure = plt.figure(figsize=(15,10))
sns.barplot(x='sex', y='mark',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.barplot(x='cp', y='mark',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.barplot(x='fbs', y='mark',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.barplot(x='restecg', y='mark',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.barplot(x='exang', y='mark',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.barplot(x='ca', y='mark',data=df1)
plt.show()




figure = plt.figure(figsize=(15,10))
sns.boxplot(x='chol',y='mark',orient='h',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.boxplot(x='rest_bps',y='mark',orient='h',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.boxplot(x='thal',y='mark',orient='h',data=df1)
plt.show()

figure = plt.figure(figsize=(15,10))
sns.boxplot(x='oldpeak',y='mark',orient='h',data=df1)
plt.show()

'''

data_x = df.drop(labels=['mark'], axis=1, inplace=False)
data_y = df['mark']


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y ,
                                    test_size = 0.3, random_state = 44, 
                                    shuffle = True)




Linear_R = LinearRegression().fit(x_train,y_train)

Logistic_R = LogisticRegression(solver="liblinear").fit(x_train,y_train)

KNN = KNeighborsClassifier().fit(x_train,y_train)

Decision_Tree = DecisionTreeClassifier(random_state=44).fit(x_train,y_train)


models = [Linear_R,Logistic_R,KNN,Decision_Tree]



total = pd.DataFrame(columns=['Score','Model'])
for model in models:
    predict = model.predict(x_test)
    cvs = cross_val_score(model,x_test,y_test).mean()
    name = model.__class__.__name__
    result = pd.DataFrame([cvs,name],columns=['Score','Model'])
    print(model)
    print(cvs)
    print('*'*20)
    total.append(result)

'''
score = pd.DataFrame(dict(Score = total, Model = models))
best_score = score.sort_values(by='Score',ascending = False)
best_model = best_score.iloc[:1]['Model']

best_predict = best_model.predict(x_test)
evaluating_score = cross_val_score(best_model,x_test,y_test).mean()


'''





