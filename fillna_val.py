import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression

# 评测指标，计算F1 score
def countF1(train, predict): 
    count = 0 # 统计预测的正确的正样本数
    for i in range(len(train)):
        if predict[i] == 1 and train[i] == 1:
            count += 1
    pre =  count * 1.0 / sum(predict) # 准确率
    recall =  count * 1.0 / sum(train) # 召回率
    return 2 * pre * recall / (pre + recall)
    
train_data = pd.read_csv('C:\\Users\\JingYi\\Desktop\\diabetes_prediction\\train_data.csv', encoding='gbk')
# 1000,85

filter_feature = ['id','label'] # 取预测值
features = []
for x in train_data.columns: # 取特征
    if x not in filter_feature:
        features.append(x)

# 缺失值填充
'''
train_data.fillna(0, inplace=True) # 填充 0
train_data.fillna(train_data.mean(),inplace=True) # 填充均值
train_data.fillna(train_data.median(),inplace=True) # 填充中位数
train_data.fillna(train_data.mode(),inplace=True) # 填充众数,该数据缺失太多众数出现为nan的情况
features_mode = {}
for f in features:
    print f,':', list(train_data[f].dropna().mode().values)
    features_mode[f] = list(train_data[f].dropna().mode().values)[0]
train_data.fillna(features_mode,inplace=True)

train_data.fillna(method='pad', inplace=True) # 填充前一条数据的值，但是前一条也不一定有值
train_data.fillna(0, inplace=True)

train_data.fillna(method='bfill', inplace=True) # 填充后一条数据的值，但是后一条也不一定有值
train_data.fillna(0, inplace=True)

for f in features: # 插值法填充
    train_data[f] = train_data[f].interpolate()
    
train_data.dropna(inplace=True)
'''


train_data.fillna(0, inplace=True) # 填充 0
train_data_x = train_data[features]
train_data_y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, random_state=1) # 划分训练集、测试集

linreg = LogisticRegression() 
linreg.fit(X_train, y_train) # 模型训练


y_pred = linreg.predict(X_train) # 模型预测
print "训练集F1:",countF1(y_train.values, y_pred)

y_pred = linreg.predict(X_test) # 模型预测
print "测试集F1:",countF1(y_test.values, y_pred)