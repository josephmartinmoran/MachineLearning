import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast_cancer_wisconsin.data')
df.replace('?', -99999, inplace=True)
# Gets rid of missing data with outlier footholder
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# tests on 20% of the data

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

#example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
#example_measures = example_measures.reshape(len(example_measures),-1)
# Predicts on any number of examples

#prediction = clf.predict(example_measures)
#print(prediction)

