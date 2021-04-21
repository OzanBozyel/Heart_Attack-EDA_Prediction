# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:38:52 2021

@author: Ozan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly_express as px
from scipy.stats.stats import pearsonr

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



df1.hist(figsize=(15,10))
plt.show()


j = 0
for i in columns:
    j = j+1
    if(j!=len(columns)):
        sns.lineplot(x='mark', y=i,data=df1)
        plt.show()
    
















