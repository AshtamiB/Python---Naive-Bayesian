# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 03:50:34 2018

@author: Ashtami
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# ------------- create a mesh to plot in ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_for_plot = np.c_[xx.ravel(), yy.ravel()]
# ------------- Create the NAÏVE Bayesian object -------------
NB = GaussianNB()
NB.fit(X, y)
Z = NB.predict(X_for_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Naive Bayesian')
plt.show()