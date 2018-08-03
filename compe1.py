import pandas as pd
import numpy as np
import sklearn
import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy import ScalarType
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

train_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\train.csv")
test_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\test.csv")

#print train_data.shape
#print test_data.shape
#tst_n = test_data.isnull().sum().sort_values(ascending=False)
#tst_r = tst_n/len(test_data)
#tst_d = pd.concat([tst_n,tst_r],axis=1,keys=['count','ratio'])
#print tst_d
num = train_data.isnull().sum().sort_values(ascending=False)
rate = num/len(train_data)
data = pd.concat([num,rate],axis=1,keys=['count','ratio'])
print data
test_data = test_data.drop(data[data['count']>1].index,axis=1)
train_data = train_data.drop(data[data['count']>1].index,axis=1)
#train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
print train_data.shape
print test_data.shape
number = [attr for attr in train_data.columns if train_data.dtypes[attr]!='object']

for c in number:
    m = train_data[c].mean()
    s = train_data[c].std()
    up = m+3*s
    bo = m-3*s
    for i in (0,len(train_data[c])-1):
        if train_data[c].loc[i] > up or train_data[c].loc[i] < bo:
            train_data = train_data.drop([i],axis=0)


train_data_x = train_data.drop('SalePrice',axis=1)
train_data_y = train_data['SalePrice']
print train_data_x.shape
quality = [attr for attr in train_data_x.columns if train_data_x.dtypes[attr]=='object']


#print quality

for c in quality:
    set_q = set(train_data_x[c])
    list_q = list(set_q)
    #print list_q
    col = range(0,len(train_data_x))

    for i in range(0,len(train_data_x)):
        for e in range(0,len(list_q)):
            try:
                if train_data_x[c].loc[i]==list_q[e]:
                    #print e
                    col[i]=e
            except Exception():
                print Exception
    #print c
    train_data_x = train_data_x.drop(c,axis=1)
    #print len(train_data_x.columns)
    train_data_x[c]=col

    test_col = range(0, len(test_data))

    for i in range(0,len(test_data)):
        for e in range(0,len(list_q)):
            try:
                if test_data[c].loc[i]==list_q[e]:
                    #print e
                    test_col[i]=float(e)
            except Exception():
                print Exception
#    print c

    test_data = test_data.drop(c,axis=1)
    test_data[c]=test_col
#    test_data[c]=test_data[c].astype('category')
#    if test_data[c].isnull().any():
#        test_data[c] = test_data[c].fillna(float(len(test_data)+1))
for c in test_data.columns:
    val = test_data[c].mean()
    if test_data[c].isnull().any():
        test_data[c] = test_data[c].fillna(val)
#print train_data_x
#print test_data

train_data_x = train_data_x.drop(train_data_x.loc[train_data['Electrical'].isnull()].index)
train_data_y = train_data_y.drop(train_data_y.loc[train_data['Electrical'].isnull()].index)

#sns.distplot(train_data_y)
#plt.show()

k=10
corrmat = train_data.corr()
#print corrmat
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
mcols = corrmat.nlargest(-k,'SalePrice')['SalePrice'].index
print mcols
print cols
cols = cols.drop('SalePrice')
cols = cols.drop('GarageArea')
cols = cols.drop('1stFlrSF')
print cols
new_tr_x = train_data_x[cols]
#new_tr_x = train_data['OverallQual','GrLivArea']
new_te_x = test_data[cols]
#new_te_x = test_data['OverallQual','GrLivArea']
#new_tr_x = new_tr_x.drop('SalePrice',axis=1)
#print new_tr_x

#cm = np.corrcoef(train_data[cols])
#print cm
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm)
#plt.show()

#quantity = [attr for attr in test_data.columns if test_data.dtypes[attr] != 'object']
#miss_cal = test_data[quality].isnull().sum().sort_values(ascending=False)
#miss = miss_cal[miss_cal>0].index
#test_data[miss] = test_data[miss].fillna(test_data[miss].mean())
#print test_data[miss].isnull().sum()
#print train_data_x.shape
#print train_data_y
#print test_data[np.isnan(test_data).any]

#line_reg = LinearRegression()#0.508
#line_reg.fit(new_tr_x,train_data_y)
re1 = list(test_data['Id'])
ind = test_data['Id']
train_data_x = train_data_x.drop('Id',axis=1)
test_data = test_data.drop('Id',axis=1)
las = LassoCV()
las.fit(new_tr_x,train_data_y)
para = {'n_alphas':[i for i in range(50,200)], 'max_iter':[j for j in range(500,2000)]}
clt = GridSearchCV(las,param_grid=para)
clt.fit(train_data_x,train_data_y)
test_data_y3 = clt.predict(test_data)
#train_data_x = preprocessing.normalize(train_data_x)
#test_data = preprocessing.normalize(test_data)
boostreg = GradientBoostingRegressor()#learning_rate=0.016,n_estimators=1000)#0.161
#boostreg.fit(new_tr_x, train_data_y)
#train_data_y = np.log1p(train_data_y)
boostreg.fit(train_data_x,train_data_y)
#my_pip = RandomForestRegressor()#0.172
#my_pip.fit(new_tr_x,train_data_y)
#print test_data
#test_data.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\test_data.csv")
#test_data_e = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\test_data.csv")
#print test_data_e.shape
#print train_data_x.shape
#test_data_y2 = line_reg.predict(new_te_x)
#test_data_y = boostreg.predict(new_te_x)
#t_y1 = boostreg.predict(train_data_x)
#t_y2 = las.predict(new_tr_x)

#t_y = [train_data_y,t_y1,t_y2]
test_data_y2 = las.predict(new_te_x)
test_data_y1 = boostreg.predict(test_data)
#test_data_y = np.expm1(test_data_y)
#test_data_y2 = my_pip.predict(new_te_x)
test_data_y = 0.2*test_data_y2+0.8*test_data_y1

re2 = list(test_data_y)

#print re1
#print re2

re_list = [re1,re2]
#print re_list
re_col = ['SalePrice']

re = pd.DataFrame(index=ind, data=test_data_y1, columns=re_col)
new_re = pd.DataFrame(index=ind, data=test_data_y3, columns=re_col)
#re_t = pd.DataFrame(data=t_y)
#re_t.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\tr_test.csv")
re.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\result.csv")
new_re.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\\grid_result.csv")


