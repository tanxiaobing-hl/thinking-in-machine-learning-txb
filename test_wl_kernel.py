# coding=utf-8


'''
Graph Kernel 包GraKeL
  https://ysig.github.io/GraKeL/dev/user_manual/introduction.html
  https://github.com/ysig/GraKeL
参考论文：
  Weisfeiler-Lehman Graph Kernels, Nino Shervashidze et al. 2011
Graph Kernel有很多种。常见的分为三类：
  基于树的，
  基于路径的，
  基于子图的
  WL核可以基于树构建，也可以基于路径构建，还可以基于子图构建。
  Informally, a kernel is a function of two objects that quantifies their similarity.
  Mathematically, it corresponds to an inner product in a reproducing kernel Hilbert space.

  Graph Kernel 都来基于 R-convolution kernel 理论： Convolution Kernels on Discrete Structures, David Haussler, 1999
  
'''
from grakel import GraphKernel, datasets

wl_kernel = GraphKernel(kernel=[{"name": "weisfeiler_lehman"}, {"name": "subtree_wl"}])
H2O = [[[[0, 1, 1], [1, 0, 0], [1, 0, 0]], {0: 'O', 1: 'H', 2: 'H'}]]
H3O = [[[[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], {0: 'O', 1: 'H', 2: 'H', 3:'H'}]]
two = [H2O[0], H3O[0]]
# k1 = wl_kernel.fit_transform(H2O)
# print(k1)
# k2 = wl_kernel.transform(H3O)
# print(k2)
# k3 = wl_kernel.fit_transform(two)
# print(k3)



# exit()

####################################################################
#
####################################################################
# 在线下载数据
mutag = datasets.fetch_dataset("MUTAG", verbose=False)
mutag_data = mutag.data

wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 5}, {"name": "subtree_wl"}], normalize=True)

split_point = int(len(mutag_data) * 0.9)
X_train, X_test = mutag_data[:split_point], mutag_data[split_point:]
wl_kernel = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "niter": 5}, {"name": "subtree_wl"}], normalize=True)

'''
K_train是训练集的特征表示
K_test是测试集的特征表示
这里是给定一种核的使用方式：基于“基分解”的思想
   Informally， 以训练集为基，计算每个样本在训练集上的坐标，从而构成一个样本的特征向量。
   所以，每个特征向量长度都是样本集的大小
   在得到特征向量后，再用SVM进行分类
在wl_kernel.fit_transform的实现过程中，这里使用了分块矩阵分开计算再合并的思想。

当然，如果给定一个数据集，给这一个核，如何进行分类呢？
 i). 可以使用上述“基分解”的思想
 ii). 也可以使用其它方法，如KNN，因为定义了核，就有了相似度的度量，也就有了距离。
 
当然，距离可以用核来表示，但距离也可以用神经网络来度量，所以就了有Metric Learning!

'''
K_train = wl_kernel.fit_transform(X_train)
K_test = wl_kernel.transform(X_test)
# K_test = wl_kernel.fit(X_train).transform(X_test)

y = mutag.target
y_train, y_test = y[:split_point], y[split_point:]

from sklearn.svm import SVC
clf = SVC(kernel='precomputed')

clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

from sklearn.metrics import accuracy_score
print("%2.2f %%" %(round(accuracy_score(y_test, y_pred)*100)))
