from PIL import Image
import numpy as np
import pandas as pd
import os

train_path = "F:\\all\salt\\train\images"
mask_path = "F:\\all\salt\\train\masks"
test_path = "F:\\all\salt\\train\images"
depth = pd.read_csv("F:\\all\salt\\depths.csv")

cols = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','depth','label']
train_data = pd.DataFrame(columns=cols)
tmp_data = np.zeros(27,dtype=int)

tmp_mat = np.zeros((105,105),dtype=int)
#print tmp_data[26]

files = os.listdir(train_path)
for f in files:
    #print f
    fileid = f.split('.png')
#    print fileid[0]
    dep = depth[depth['id']==fileid[0]]
    train_png = train_path+'\\'+f
    train_mask = mask_path+'\\'+f
    img = Image.open(train_png)
    img = img.convert('L')
    mat = np.array(img)
    mask = Image.open(train_mask)
    mask = mask.convert('L')
    label = np.array(mask)
    #print mat.size
    #print mat
    #print label
    #print dep['z']
    for i in range(0,101):
        for j in range(0,101):
            tmp_mat[i+2][j+2] = mat[i][j]
            print [i,j]

    for i in range(2,103):
        for j in range(2,103):
            tmp_data[0] = tmp_mat[i - 2][j - 2]
            tmp_data[1] = tmp_mat[i - 2][j - 1]
            tmp_data[2] = tmp_mat[i - 2][j]
            tmp_data[3] = tmp_mat[i - 2][j + 1]
            tmp_data[4] = tmp_mat[i - 2][j + 2]
            tmp_data[5] = tmp_mat[i - 1][j - 2]
            tmp_data[6] = tmp_mat[i - 1][j - 1]
            tmp_data[7] = tmp_mat[i - 1][j]
            tmp_data[8] = tmp_mat[i - 1][j + 1]
            tmp_data[9] = tmp_mat[i - 1][j + 2]
            tmp_data[10] = tmp_mat[i][j - 2]
            tmp_data[11] = tmp_mat[i][j - 1]
            tmp_data[12] = tmp_mat[i][j]
            tmp_data[13] = tmp_mat[i][j + 1]
            tmp_data[14] = tmp_mat[i][j + 2]
            tmp_data[15] = tmp_mat[i + 1][j - 2]
            tmp_data[16] = tmp_mat[i + 1][j - 1]
            tmp_data[17] = tmp_mat[i + 1][j]
            tmp_data[18] = tmp_mat[i + 1][j + 1]
            tmp_data[19] = tmp_mat[i + 1][j + 2]
            tmp_data[20] = tmp_mat[i + 2][j - 2]
            tmp_data[21] = tmp_mat[i + 2][j - 1]
            tmp_data[22] = tmp_mat[i + 2][j]
            tmp_data[23] = tmp_mat[i + 2][j + 1]
            tmp_data[24] = tmp_mat[i + 2][j + 2]
            tmp_data[25] = dep['z']
            tmp_data[26] = label[i-2][j-2]
            d = pd.DataFrame(tmp_data)
            print tmp_data
            print d
            train_data.append(d)

    print train_data
    train_data.to_csv("F:\\all\salt\\train\\tmp.csv")
            #train_data.ix[i*10+j].append("ddd")







#img = Image.open("F:\\all\salt\\train\images\\000e218f21.png")


#img = img.convert('L')

#mat = np.array(img)

#print mat