import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

#讀取資料
file = open(r'C:/Users/kitty/tensorflow/decision_tree/input_data.csv','r')
data = csv.reader(file)
#資料前處理
target = []
temp_data = []
temp_target = []
input_data = []
temp_all_feature = []
all_feature = []
x = []
for row in data:
    temp_data.append(row[:4])
    temp_target.append(row[-1])
    temp_all_feature.append(row)
for row in temp_data[1:]:
    for i in row:
        x.append(float(i))
    input_data.append(x)
    x = []

for i in temp_target[1:]:
    target.append(float(i))
feature_name = temp_data[0]
target_name = [temp_target[0]]
for i in temp_all_feature[0]:
    all_feature.append(i)

x = pd.DataFrame(input_data,columns=feature_name)
y = pd.DataFrame(target,columns=target_name)
predict_data = pd.concat([x,y],axis = 1)
predict_data = predict_data[all_feature]


#建立決策樹
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

X_train,X_test,y_train,y_test = train_test_split(predict_data[feature_name],predict_data[target_name],test_size = 0.3,random_state = 0)

mae = 0
depth = 1
sample_split = 50
sample_leaf =50
for i in range(5):
    print("max_depth:",depth)
    for j in range(1):
        print("min_samples_split:",sample_split)
        for z in range(1):
            tree = DecisionTreeClassifier(criterion = "entropy",max_depth = depth,random_state = 0,min_samples_split=sample_split,min_samples_leaf=sample_leaf)
            tree.fit(X_train,y_train)
            error = 0
            #計算準確率&MAE
            for i,v in enumerate(tree.predict(X_test)):        
                e = abs(y_test[target_name[0]].values[i]-v)
                mae = mae + e  
                if v!=y_test[target_name[0]].values[i]:
                    error += 1
            mae = mae/len(X_test)
            print("min_samples_leaf:",sample_leaf,"mean_accuracy: ",round(tree.score(X_test,y_test[target_name[0]]),2) , "error:" , error,"MAE: ",round(mae,2))
            sample_leaf = sample_leaf + 1
            mae = 0 
        sample_split = sample_split + 10
        sample_leaf = 50
    sample_split =10
    depth += 1
