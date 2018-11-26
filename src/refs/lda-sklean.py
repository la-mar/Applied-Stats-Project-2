#! https://www.python-course.eu/linear_discriminant_analysis.php
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, PrincipleComponenetAnalysis
# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('data/Wine.txt',sep=',',names=['target','Alcohol','Malic_acid','Ash','Akcakinity','Magnesium','Total_pheonols','Flavanoids','Nonflavanoids','Proanthocyanins','Color_intensity','Hue','OD280','Proline'])
X = df.iloc[:,1:].copy()
target = df['target'].copy()
train_x, test_x, train_y, test_y = train_test_split(X,target,test_size=0.3,random_state=0) 
# 1. Instantiate the method and fit_transform the algotithm
LDA = LinearDiscriminantAnalysis(n_components=2) # The n_components key word gives us the projection to the n most discriminative directions in the dataset. We set this parameter to two to get a transformation in two dimensional space.  
result = LDA.fit_transform(train_x,train_y)
print(result.shape)
# PLot the transformed data
markers = ['s','x','o']
colors = ['r','g','b']
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for l,m,c in zip(np.unique(train_y),markers,colors):
    ax0.scatter(result[:,0][train_y==l],result[:,1][train_y==l],c=c,marker=m)