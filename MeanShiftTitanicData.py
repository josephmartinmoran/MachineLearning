import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
import xlrd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name 
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
# print(df.head())
original_df = pd.DataFrame.copy(df)

df.drop(['body', 'name'], axis=1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)


# print(df.head())

def to_numeric_data(df):
    column = df.columns.values
    for columns in column:
        if df[columns].dtype != np.int64 and df[columns].dtype != np.float64:
            df[columns] = pd.factorize(np.array(df[columns]))[0]
    return df


df = to_numeric_data(df)

# print(df.head())

# df.drop(['ticket'], 1, inplace=True)
# df.drop(['boat'], 1, inplace=True)
df.drop(columns=['sex'], inplace=True)
X = np.array(df.drop(columns=['survived']).astype(float))

X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df.loc[i, 'cluster_group'] = labels[i]
n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
print(original_df[(original_df['cluster_group']==1)].describe())
print(original_df[(original_df['cluster_group']==2)].describe())
print(original_df[(original_df['cluster_group']==0)].describe())

cluster_0 = original_df[(original_df['cluster_group']==0)]
cluster_0_fc = cluster_0[(cluster_0['pclass']==1)]
print(cluster_0_fc.describe())

