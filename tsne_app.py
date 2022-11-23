from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import time

import matplotlib

matplotlib.use('TkAgg')

data = pd.read_csv("mnist_test.csv", delimiter=',')

start_time = time.time()

D = data.drop(['label'], axis=1)
scaler = preprocessing.MinMaxScaler()
D = pd.DataFrame(scaler.fit_transform(D), columns=D.columns)

T = TSNE(n_components=2, perplexity=50, random_state=123)
TSNE_features = T.fit_transform(D)
DATA = D.copy()
DATA['x'] = TSNE_features[:, 0]
DATA['y'] = TSNE_features[:, 1]
finish_time = time.time()

fig = plt.figure()
sns.scatterplot(x='x', y='y', hue=data['label'], data=DATA, palette='bright')
plt.show()

print(finish_time-start_time)
