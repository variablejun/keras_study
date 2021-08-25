import numpy as np
import matplotlib.pyplot  as plt

def scaled_elu(x, a=1.6733, t=1.0507):
     result = []
     for i in x:
        if i<0:
            i = t*(a*(np.exp(i)-1))
        result.append(i)
     return result

x = np.arange(-5,5,0.1)
y = scaled_elu(x)

plt.plot(x,y)
plt.grid()
plt.show()
