import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('very.csv')


# In[3]:


def normalization(L):
    min_value = L.min()
    max_value = L.max()
    d = max_value - min_value
    L_new = []
    for each in L:
        L_new.append((each-min_value)/d)
    return L_new


# In[4]:


for i in data:
    if type(data[i][2]) != str:
        data[i] = pd.Series(normalization(data[i]))


# In[5]:


import random
def split(l,k=5):
    L = []
    for i in range(l):
        L.append(i)
    random.shuffle(L)
    x = 0
    ct =int(l/5)
    y = x+ct
    res = []
    for i in range(k):
        res.append(L[x:y])
        x = x+ct
        y = y+ct
        if i == k-2:
            y = l
        
    return res
        


# In[6]:


s = split(len(data))
M=[]
N=[]
for i in range(len(s)):
    data_train = data.drop(s[i],axis=0)
    data_test = data.drop(list(data_train.index),axis=0)
    data_train.reset_index(inplace=True)
    data_train.drop('index',axis=1,inplace=True)
    data_test.reset_index(inplace=True)
    data_test.drop('index',axis=1,inplace=True)
    M.append(data_train)
    M.append(data_test)
    N.append(M)
    M=[]
    data_train=None
    data_test=None
k_fold = N


# In[7]:


def sqd(l1,l2):
    res = 0
    for i in range(len(l1)):
        x = l1[i]-l2[i]
        x = abs(x)
        res = res + x
        if res == 0:
            res = 50
    return res


# In[8]:


def mean(l):
    mean_l=[]
    tmp = 0
    x=0
    for i in range(len(l[0])):
        for j in range (len(l)):
            x = x + l[j][i]
        x = x / len(l)
        mean_l.append(x)
        x = 0
    return (mean_l)


# In[9]:


def location(d,m):
    for i in range(len(d)):
        if m in d[i]:
            return i,d[i].index(m)


# In[10]:


def knn(train,test,column='Class',k=3):
    data_test = test.drop(['name','Class'],axis=1)
    data_train = train.drop(['name','Class'],axis=1)
    test_l = data_test.values
    train_l = data_train.values
    Classes=[]
    Dist = []
    for each in test_l:
        for j in range(len(train_l)):
            Dist.append([sqd(each,train_l[j]),j])
        Dist.sort()
        Dist = Dist[:k]
        c=[]
        for i in range(k):
            c.append(Dist[i][1])
        values, counts = np.unique(c, return_counts=True)
        values = list(values)
        counts = list(counts)
        Classes.append(train.Class[values[counts.index(max(counts))]])
        Dist=[]
    print(test.Class)    #original data
    print(np.array(Classes))   # classificated
    res=0
    for i in range(len(test.Class)):
        if test.Class[i]==Classes[i]:
            res = res + 1
    res = res/(len(test))
    return  res


# In[11]:


max_accuracy = 0
tmp_i = 0
for i in range(len(k_fold)):
    tmp = knn(k_fold[i][0],k_fold[i][1],k=1)
    if tmp > max_accuracy:
        max_accuracy = tmp
        tmp_i = i
print('k_fold:' + str(tmp_i))
print('accuracy:' + str(max_accuracy))
