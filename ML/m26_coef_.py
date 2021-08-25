#coeffition 계수

x = [-3,31,-11,4,0,22,-2,-5,-25,-14]
y = [-3,65,-19,11,3,47,-1,-7,-47,-25]
'''
import matplotlib.pyplot as plt

plt.plot(x,y)
plt.show()
'''
import pandas as pd

df = pd.DataFrame({'x':x,'y':y})
print(df.shape)

x_train = df.loc[:,'x']
y_train = df.loc[:,'y']
print(x_train.shape)
print(x_train.shape)

x_train = x_train.values.reshape(len(x_train),1)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)

score = model.score(x_train,y_train)
print('acc',score)

print('기울기',model.coef_)

print('절편',model.intercept_)

'''
acc 1.0
기울기 [2.]
절편 3.0
'''