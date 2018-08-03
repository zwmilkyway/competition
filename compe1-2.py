import pandas as pd
import numpy as np
import sklearn
import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import normalize
from scipy import ScalarType
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

train_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\train.csv")
test_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\test.csv")


num = train_data.isnull().sum().sort_values(ascending=False)
rate = num/len(train_data)
data = pd.concat([num,rate],axis=1,keys=['count','ratio'])
#print data
test_data = test_data.drop(data[data['count']>1].index,axis=1)
train_data = train_data.drop(data[data['count']>1].index,axis=1)
#train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)

te_num = test_data.isnull().sum().sort_values(ascending=False)
te_rate = te_num/len(test_data)
t_data = pd.concat([te_num,te_rate],axis=1,keys=['count','ratio'])
#print t_data
#print train_data.shape
#print test_data.shape
#print train_data
train_data = train_data.drop('Id',axis=1)
#tmp = test_data
#test_data = test_data.drop('Id',axis=1)



for c in test_data.columns:
    if test_data.dtypes[c]!=object:
        val = test_data[c].mean()
        if test_data[c].isnull().any():
            test_data[c] = test_data[c].fillna(val)
    if test_data.dtypes[c]==object:
        count = Counter(test_data[c])
        list_re = count.most_common(1)
        v = list_re[0][0]
        if test_data[c].isnull().any():
            test_data[c] = test_data[c].fillna(v)

#    val = test_data[c].mean()
#    if test_data[c].isnull().any():
#        test_data[c] = test_data[c].fillna(val)
quality = [attr for attr in train_data.columns if train_data.dtypes[attr]=='object']

#print quality

for c in quality:
    set_q = set(train_data[c])
    list_q = list(set_q)
    #print list_q
    col = range(0,len(train_data[c]))

    for i in range(0,len(train_data[c])):
        for e in range(0,len(list_q)):
            try:
                if train_data[c].loc[i]==list_q[e]:
                    #print e
                    col[i]=e
            except Exception():
                print Exception

    train_data = train_data.drop(c,axis=1)
    train_data[c]=col

    test_col = range(0, len(test_data[c]))

    for i in range(0,len(test_data[c])):
        for e in range(0,len(list_q)):
            try:
                if test_data[c].loc[i]==list_q[e]:
                    test_col[i]=float(e)
            except Exception():
                print Exception
    test_data = test_data.drop(c, axis=1)
    test_data[c] = test_col

train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)

col_name = ['SalePrice']
new_re = pd.DataFrame(columns=col_name)
set_zoning = set(train_data['MSZoning'])
list_zoning = list(set_zoning)
for e in list_zoning:
    print e
    trd = train_data[train_data['MSZoning']==e]

    #print trd
    tsd = test_data[test_data['MSZoning']==e]
    ind = tsd['Id']
    train_data_y = trd['SalePrice']
    trd = trd.drop('SalePrice',axis=1)
    trd = trd.drop('MSZoning',axis=1)
    tsd = tsd.drop('MSZoning',axis=1)
    tsd = tsd.drop('Id',axis=1)

    bootreg = GradientBoostingRegressor()
    bootreg.fit(trd,train_data_y)
    test_y = bootreg.predict(tsd)

    re = pd.DataFrame(index=ind,data=test_y,columns=col_name)
    new_re = new_re.append(re)
    print re

new_re.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\new_re.csv",index_label='Id')


