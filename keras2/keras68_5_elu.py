import numpy as np
import matplotlib.pyplot  as plt

def elu(x):
     return np.maximum(0.01*x,x)

x = np.arange(-5,5,0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()
