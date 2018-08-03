import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer

test_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\test.csv")
train_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\train.csv")

train_data_tr = train_data
#test_data_te = test_data


#print test_data.describe()
data0 = train_data_tr[train_data['label']==0].drop('label',axis=1)
data1 = train_data_tr[train_data['label']==1].drop('label',axis=1)
data2 = train_data_tr[train_data['label']==2].drop('label',axis=1)
data3 = train_data_tr[train_data['label']==3].drop('label',axis=1)
data4 = train_data_tr[train_data['label']==4].drop('label',axis=1)
data5 = train_data_tr[train_data['label']==5].drop('label',axis=1)
data6 = train_data_tr[train_data['label']==6].drop('label',axis=1)
data7 = train_data_tr[train_data['label']==7].drop('label',axis=1)
data8 = train_data_tr[train_data['label']==8].drop('label',axis=1)
data9 = train_data_tr[train_data['label']==9].drop('label',axis=1)

bd0 = preprocessing.binarize(data0)
bd1 = preprocessing.binarize(data1)
bd2 = preprocessing.binarize(data2)
bd3 = preprocessing.binarize(data3)
bd4 = preprocessing.binarize(data4)
bd5 = preprocessing.binarize(data5)
bd6 = preprocessing.binarize(data6)
bd7 = preprocessing.binarize(data7)
bd8 = preprocessing.binarize(data8)
bd9 = preprocessing.binarize(data9)
#Binarizer(threshold=0).fit(data0)
da0 = pd.DataFrame(columns=data0.columns,data=bd0)
da1 = pd.DataFrame(columns=data0.columns,data=bd1)
da2 = pd.DataFrame(columns=data0.columns,data=bd2)
da3 = pd.DataFrame(columns=data0.columns,data=bd3)
da4 = pd.DataFrame(columns=data0.columns,data=bd4)
da5 = pd.DataFrame(columns=data0.columns,data=bd5)
da6 = pd.DataFrame(columns=data0.columns,data=bd6)
da7 = pd.DataFrame(columns=data0.columns,data=bd7)
da8 = pd.DataFrame(columns=data0.columns,data=bd8)
da9 = pd.DataFrame(columns=data0.columns,data=bd9)
da0.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\0.csv")
print 'done'

temp0 = da0.mean()
temp1 = da1.mean()
temp2 = da2.mean()
temp3 = da3.mean()
temp4 = da4.mean()
temp5 = da5.mean()
temp6 = da6.mean()
temp7 = da7.mean()
temp8 = da8.mean()
temp9 = da9.mean()


re = pd.DataFrame(data=np.random.randint(low=0,high=10,size=(len(test_data),1)),columns=['Label'])
for c in range(0, len(test_data)):
    tmp_te = preprocessing.binarize([test_data.ix[c]])
    test_data_te = tmp_te[0]

    print test_data_te
    d0 = np.linalg.norm(test_data_te - temp0)
    d1 = np.linalg.norm(test_data_te - temp1)
    d2 = np.linalg.norm(test_data_te - temp2)
    d3 = np.linalg.norm(test_data_te - temp3)
    d4 = np.linalg.norm(test_data_te - temp4)
    d5 = np.linalg.norm(test_data_te - temp5)
    d6 = np.linalg.norm(test_data_te - temp6)
    d7 = np.linalg.norm(test_data_te - temp7)
    d8 = np.linalg.norm(test_data_te - temp8)
    d9 = np.linalg.norm(test_data_te - temp9)

    m = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
    dm = min(m)
    #print dm
    if (dm == d0):
        re.ix[c]=0
    if (dm == d1):
        re.ix[c]=1
    if (dm == d2):
        re.ix[c]=2
    if (dm == d3):
        re.ix[c]=3
    if (dm == d4):
        re.ix[c]=4
    if (dm == d5):
        re.ix[c]=5
    if (dm == d6):
        re.ix[c]=6
    if (dm == d7):
        re.ix[c]=7
    if (dm == d8):
        re.ix[c]=8
    if (dm == d9):
        re.ix[c]=9

re.index += 1
re.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\re.csv",index_label='ImageId',index=True)




