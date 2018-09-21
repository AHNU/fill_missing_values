# fill_missing_values
常见的缺失值填充方法有填充默认值、均值、众数、KNN填充、以及把缺失值作为新的label通过模型来预测等方式，为了介绍这几种填充方法的使用以及填充效果，本文将在真实数据集上进行简单比较。

fillna_val.py : 实现了常见的默认值、均值、众数等填充方式。
fancyimpute_knn.py :实现了KNN填充方式。
fillna_rf.py :实现了采用随机森林模型预测缺失值的填充方式。
