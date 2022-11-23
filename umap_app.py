from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import time

import umap.umap_ as umap

import matplotlib

matplotlib.use('TkAgg')

data = pd.read_csv("mnist_test.csv", delimiter=',')

start_time = time.time()

D = data.drop(['label'], axis=1)
scaler = preprocessing.MinMaxScaler()
D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)
DATA = D.copy()

reducer = umap.UMAP(n_neighbors=50, min_dist=0.6, random_state=123)
embedding = reducer.fit_transform(DATA)
DATA['x'] = embedding[:, 0]
DATA['y'] = embedding[:, 1]
finish_time = time.time()
print(finish_time - start_time)

fig = plt.figure()
sns.scatterplot(x='x', y='y', hue=data['label'], data=DATA, palette='bright')
plt.show()
