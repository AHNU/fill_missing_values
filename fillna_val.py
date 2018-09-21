import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression

# ����ָ�꣬����F1 score
def countF1(train, predict): 
    count = 0 # ͳ��Ԥ�����ȷ����������
    for i in range(len(train)):
        if predict[i] == 1 and train[i] == 1:
            count += 1
    pre =  count * 1.0 / sum(predict) # ׼ȷ��
    recall =  count * 1.0 / sum(train) # �ٻ���
    return 2 * pre * recall / (pre + recall)
    
train_data = pd.read_csv('C:\\Users\\JingYi\\Desktop\\diabetes_prediction\\train_data.csv', encoding='gbk')
# 1000,85

filter_feature = ['id','label'] # ȡԤ��ֵ
features = []
for x in train_data.columns: # ȡ����
    if x not in filter_feature:
        features.append(x)

# ȱʧֵ���
'''
train_data.fillna(0, inplace=True) # ��� 0
train_data.fillna(train_data.mean(),inplace=True) # ����ֵ
train_data.fillna(train_data.median(),inplace=True) # �����λ��
train_data.fillna(train_data.mode(),inplace=True) # �������,������ȱʧ̫����������Ϊnan�����
features_mode = {}
for f in features:
    print f,':', list(train_data[f].dropna().mode().values)
    features_mode[f] = list(train_data[f].dropna().mode().values)[0]
train_data.fillna(features_mode,inplace=True)

train_data.fillna(method='pad', inplace=True) # ���ǰһ�����ݵ�ֵ������ǰһ��Ҳ��һ����ֵ
train_data.fillna(0, inplace=True)

train_data.fillna(method='bfill', inplace=True) # ����һ�����ݵ�ֵ�����Ǻ�һ��Ҳ��һ����ֵ
train_data.fillna(0, inplace=True)

for f in features: # ��ֵ�����
    train_data[f] = train_data[f].interpolate()
    
train_data.dropna(inplace=True)
'''


train_data.fillna(0, inplace=True) # ��� 0
train_data_x = train_data[features]
train_data_y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, random_state=1) # ����ѵ���������Լ�

linreg = LogisticRegression() 
linreg.fit(X_train, y_train) # ģ��ѵ��


y_pred = linreg.predict(X_train) # ģ��Ԥ��
print "ѵ����F1:",countF1(y_train.values, y_pred)

y_pred = linreg.predict(X_test) # ģ��Ԥ��
print "���Լ�F1:",countF1(y_test.values, y_pred)