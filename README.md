# Homework_3

import numpy as np
import pandas as pd

raw_data= pd.read_csv('HW3Data.csv', sep=",", header = None)
raw_data.head()

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=62, n_features=2000, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1, random_state=0)
svc = SVC(kernel="linear")

rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (# of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

from sklearn.cross_validation import train_test_split

X_new = rfecv.transform(X)

X = X_new
y = raw_data.iloc[: , 0]
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)
print('Train set size %s' % (X_train.shape,))
print('Test set size %s' % (X_test.shape,))

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(estimator=svc, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
