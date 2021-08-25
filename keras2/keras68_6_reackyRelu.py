import numpy as np
import matplotlib.pyplot  as plt

def reackyRelu(x):
     return np.maximum(0.1*x,x)

x = np.arange(-5,5,0.1)
y = reackyRelu(x)

plt.plot(x,y)
plt.grid()
plt.show()
