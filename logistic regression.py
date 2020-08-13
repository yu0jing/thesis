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


#建構邏輯迴歸
from sklear.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(solver='liblinear', random_state=0,).fit(input_data, target)
model.classes_
model.intercept_
model.coef_

# 計算準確率
mae = 0
for i,v in enumerate(model.predict(input_data)):
    e = abs(target[i]-v)
    mae = mae + e 
mae = round(mae/len(input_data),2)
print(mae)
model.score(input_data, target)
