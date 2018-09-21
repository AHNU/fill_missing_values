import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

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

new_label = 'SNP46'
new_features = []
for f in features:
    if f != new_label:
        new_features.append(f)
        
new_train_x = train_data[train_data[new_label].isnull()==False][new_features]
new_train_x.fillna(new_train_x.mean(), inplace=True) # 其他列填充均值
new_train_y = train_data[train_data[new_label].isnull()==False][new_label]

new_predict_x = train_data[train_data[new_label].isnull()==True][new_features]
new_predict_x.fillna(new_predict_x.mean(), inplace=True) # 其他列填充均值
new_predict_y = train_data[train_data[new_label].isnull()==True][new_label]

rfr = RandomForestRegressor(random_state=666, n_estimators=10, n_jobs=-1)
rfr.fit(new_train_x, new_train_y)
new_predict_y = rfr.predict(new_predict_x)

new_predict_y = pd.DataFrame(new_predict_y, columns=[new_label], index=new_predict_x.index)
new_predict_y = pd.concat([new_predict_x, new_predict_y], axis=1)
new_train_y = pd.concat([new_train_x, new_train_y], axis=1)
new_train_data = pd.concat([new_predict_y,new_train_y]) 

train_data_x = new_train_data[features]
train_data_y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, random_state=1) # 划分训练集、测试集


linreg = LogisticRegression() 
linreg.fit(X_train, y_train) # 模型训练


y_pred = linreg.predict(X_train) # 模型预测
print "训练集F1:",countF1(y_train.values, y_pred)

y_pred = linreg.predict(X_test) # 模型预测
print "测试集F1:",countF1(y_test.values, y_pred)