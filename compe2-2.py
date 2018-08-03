import pandas as pd
import numpy as np
import sklearn.preprocessing as sp

test_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\test.csv")
train_data = pd.read_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\train.csv")

train_data_x = train_data.drop('label',axis=1)

re = pd.DataFrame(data=np.random.randint(low=0,high=10,size=(len(test_data),1)),columns=['Label'])
dis_min = pd.DataFrame(data=np.random.randint(low=10000,high=10001,size=(len(test_data),1)))
dis = pd.DataFrame(data=np.random.randint(low=10000,high=10001,size=(len(test_data),1)))
for i in range(0,len(train_data_x)):
    tr_x = sp.binarize([train_data_x.ix[i]])
    train_x = tr_x[0]
    for c in range(0,len(test_data)):
        te_x = sp.binarize([test_data.ix[c]])
        test_x = te_x[0]
        dis.ix[c] = np.linalg.norm(test_x-train_x)
        dis_num = list(dis.ix[c])
        dis_min_num = list(dis_min.ix[c])

        r = dis_num[0] - dis_min_num[0]
        if r < 0:
            dis_min.ix[c] = dis.ix[c]
            re.ix[c] = train_data['label'].ix[i]
            print re.ix[c]

re.index += 1
re.to_csv("C:\Users\Administrator.ZGC-20120912ILY\Desktop\\all\digital-recog\\re.csv",index_label='ImageId',index=True)
