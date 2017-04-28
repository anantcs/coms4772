
# coding: utf-8

# In[1]:

from sklearn.decomposition import *
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from mysklearn import NewMiniBatchDictionaryLearning

# A function to read and show images from mnist

# In[2]:

def read(dataset = "training", path = "../dataset/"):
    if dataset is "training":
        fname_img = os.path.join(path+"/train", 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path+"train", 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path+"test", 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path+"test", 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


# Training to get the dictionaries

# In[3]:

X = []
y = []
for label,img in read("training"):
    X.append(img.reshape(1,784)[0])
    y.append(label)

X = np.array(X)
t0 = time()
print("Calling New Mini Batch Dictionary Learning")
vader = NewMiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500,transform_algorithm='fista')
print("Done with New Mini Batch Dictionary")
V = vader.fit(X).components_
print("Done with V", V.shape)



# We transform the image space into representation space

# In[4]:

U = vader.transform(X).T
print U.shape


# We then train a Knn classifier on the transformed space and test images

# In[5]:

from sklearn.neighbors import KNeighborsClassifier

X_test = []
y_test = []

for label,img in read("testing"):
    X_test.append(img.reshape(1,784)[0])
    y_test.append(label)

#X_test = vader.transform(X_test)
nbrs = KNeighborsClassifier(n_neighbors=3)
nbrs.fit(U,y)
my_x = vader.transform(X_test).T


# In[6]:

preds = nbrs.predict(my_x)


# In[16]:

error = 0
for i in xrange(len(y_test)):
    #print preds[i],y_test[i]
    if preds[i]!=y_test[i]:
        #print preds[i],y_test[i]
        #print X_test[i].shape
        #show(X_test[i].reshape(28,28))
        error+=1
print error/100.0


# ## PCA

# In[30]:
'''
anakin = PCA(n_components = 100)
P = anakin.fit(X).components_
J = anakin.transform(X)


# In[31]:

nbrs.fit(J,y)
my_x = anakin.transform(X_test)


# In[32]:

preds = nbrs.predict(my_x)


# In[33]:

error = 0
for i in xrange(len(y_test)):
    #print preds[i],y_test[i]
    if preds[i]!=y_test[i]:
        #print preds[i],y_test[i]
        #print X_test[i].shape
        #show(X_test[i].reshape(28,28))
        error+=1
print error


# In[22]:

print(J.shape)


# In[25]:

print(len(y_test))

'''
# In[ ]:



