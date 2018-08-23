import sklearn
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor


train_data = pd.read_csv("F://all//train.csv")

train_data_y = train_data['SalePrice']
#train_data_x = train_data.drop('SalePrice',axis=1)
#train_data_x = train_data_x.drop('Id',axis=1)

#num = train_data_x.isnull().sum().sort_values(ascending=False)
#rate = num/len(train_data_x)

#sheet = pd.concat([num,rate],axis=1,keys=['num','rate'])

#print sheet

#train_data_x = train_data_x.drop(sheet[sheet['num']>1].index,axis=1)

cormat = train_data.drop('Id',axis=1).corr()
k=10
cols = cormat.nlargest(k,'SalePrice')['SalePrice'].index
cor_data_l = cormat.nlargest(k,'SalePrice')['SalePrice']
print cols
print cor_data_l
mcols = cormat.nsmallest(k,'SalePrice')['SalePrice'].index
cor_data_s = cormat.nsmallest(k,'SalePrice')['SalePrice']
#print mcols
#print cor_data_s
train_data_x = train_data[cols]
train_data_x = train_data_x.drop('SalePrice',axis=1)
train_data_x = train_data_x.drop('GarageArea',axis=1)
#train_data_x = train_data_x.drop('GarageYrBlt',axis=1)
#print train_data_x

#num = train_data_x.isnull().sum().sort_values(ascending=False)
#rate = num/len(train_data_x)

#sheet = pd.concat([num,rate],axis=1,keys=['num','rate'])
#print sheet
#print train_data_x
#print train_data_y


test_data = pd.read_csv("F://all//test.csv")
#test_data_x = test_data.drop(sheet[sheet['num']>1].index,axis=1)
test_cols = cols.drop('SalePrice')
test_data_x = test_data[test_cols]
test_data_x = test_data_x.drop('GarageArea',axis=1)
#test_data_x = test_data_x.drop('GarageYrBlt',axis=1)

#num = test_data_x.isnull().sum().sort_values(ascending=False)
#rate = num/len(test_data_x)
#sheet = pd.concat([num,rate],axis=1,keys=['num','rate'])
#print sheet
#print test_data
for c in test_data_x.columns:
    train_val = train_data_x[c].mean()
    test_val = test_data_x[c].mean()
    if test_data_x[c].isnull().any():
        test_data_x[c] = test_data_x[c].fillna(test_val)
    if train_data_x[c].isnull().any():
        train_data_x[c] = train_data_x[c].fillna(train_val)

las = Lasso(normalize=True)
para = {'alpha':[float(i)/10 for i in range(1,500)],'max_iter':[i for i in range(500,5000)]}
clt = GridSearchCV(las,param_grid=para)
clt.fit(train_data_x,train_data_y)
test_data_y = clt.predict(test_data_x)

#ada = AdaBoostRegressor(n_estimators=400)
#ada.fit(train_data_x,train_data_y)
#test_data_y = ada.predict(test_data_x)

re = pd.DataFrame(data=test_data_y,columns=['SalePrice'],index=test_data['Id'])

re.to_csv('F://all//result.csv',index_label='Id')
best_parameters = clt.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)