from sklearn.decomposition import *
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time

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

X = []

for label,img in read("training"):
   #print img.reshape(1,784)[0]
   #raw_input()
   X.append(img.reshape(1,784)[0])
#print len(X)
X = np.array(X)
t0 = time()
#print X.shape
vader = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
V = vader.fit(X).components_
D = vader.transform(X)
dt = time() - t0
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(D):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(28,28), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(X)),
             fontsize=16)
plt.show()