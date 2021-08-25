import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])


print(dataset.target_names) #['setosa' 'versicolor' 'virginica']

x = dataset.data()
y = dataset.target()

df = pd.DateFrame(x, columns = dataset['feature_names'])
print(df)

# y컬럼 추가
df['target'] = y


# 상관계수 맵
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),square=True,annot=True,cbar=True)

plt.show()
