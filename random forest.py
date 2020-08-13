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

#建構隨機森林
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test = train_test_split(predict_data[feature_name],predict_data[target_name],test_size = 0.3,random_state = 0)
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=3,n_jobs=1)
forest.fit(X_train,y_train[target_name[0]].values)

import time
tStart = time.time()
estimators =1
state =4
mae = 0
for i in range(100):
    for j in range(1):
        forest = RandomForestClassifier(criterion='entropy', n_estimators=estimators,random_state=state,n_jobs=1,max_depth=None,min_samples_leaf=1, min_samples_split=2)
        forest.fit(X_train,y_train[target_name[0]].values)
        error = 0
        for i,v in enumerate(forest.predict(X_test)):
            e = abs(y_test[target_name[0]].values[i]-v)
            if v!=y_test[target_name[0]].values[i]:
                error += 1
            mae = mae + e 
        mae = mae/len(X_test)
        print("estimators:",estimators, "state:",state,"mean_accuracy:",round(forest.score(X_test,y_test[target_name[0]]),2) , "error:" , error,"MAE: ",round(mae,2))
        state += 1
    state = 1
    estimators +=1
    mae = 0
tEnd = time.time()
print("It cost %f sec" %(tEnd - tStart))
print(tEnd - tStart)
