import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute # https://stackoverflow.com/questions/51695071/pip-install-ecos-error-microsoft-visual-c-14-0-is-required

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

train_data_x = train_data[features]
train_data_x = pd.DataFrame(KNN(k=6).fit_transform(train_data_x), columns=features)
train_data_y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, random_state=1) # 划分训练集、测试集

linreg = LogisticRegression() 
linreg.fit(X_train, y_train) # 模型训练


y_pred = linreg.predict(X_train) # 模型预测
print ("训练集",countF1(y_train.values, y_pred))

y_pred = linreg.predict(X_test) # 模型预测
print ("测试集",countF1(y_test.values, y_pred))

