import numpy as np
a = np.array(range(1,11))
size = 5
def split_x(dataset, size):
     aaa=[]
     for i in range(len(dataset) - size + 1): # 행개수 결정 6행
          subset = dataset[i : (i + size)]
          aaa.append(subset)
     return np.array(aaa)


dataset = split_x(a, size)

print(dataset)

x = dataset[:, :4]
y = dataset[:,4]

print(x)
print(y)