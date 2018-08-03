import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from sklearn.ensemble import GradientBoostingClassifier

#test_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\test.csv")
train_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\train.csv")

train_data_x = train_data.drop('label',axis=1)
train_data_y = train_data['label']

#tr_x = train_data_x
#tr_y = train_data_y
#te_x = test_data

#for i in range(0,len(train_data_x)):
#    train_data_x.ix[i] = sp.binarize([train_data_x.ix[i]])
#for i in range(0,len(test_data)):
#    test_data.ix[i] = sp.binarize([test_data.ix[i]])

gb = GradientBoostingClassifier(n_estimators=500)

gb.fit(train_data_x[:40000],train_data_y[:40000])
#re_data = gb.predict(test_data)

test_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\test.csv")
re_data = gb.predict(test_data)
re = pd.DataFrame(data=re_data, columns=['label'])
re.index += 1
re.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\re.csv",index_label='ImageId',index=True)

