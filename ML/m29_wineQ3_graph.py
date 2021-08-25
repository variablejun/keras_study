import numpy as np
import pandas as pd
from xgboost import XGBClassifier
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0)

import matplotlib.pyplot as plt

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)
#count_data.plot()
plt.bar(count_data.index, count_data)
plt.show()
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''