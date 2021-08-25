from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
import random


x = np.array(range(100)) # 0 ~ 99
y = np.array(range(1,101))  # 1 ~ 100


x_train = x[:70]

y_train = y[:70]

x_test = x[-30:]
y_test = y[70:]



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
test_size = 0.7, random_state=66) # ramdom 값이 반복할 때 마다 다르기 때문에 값을 동일하게 뽑아주기 위해서 

print(x_test)
print(y_test)


"""
결정계수 (R2) : 회기모델의 지표, max= 1 , min = 0 
sklearn shuffle은 기본적으로 True

train

[76 31 15 48 22 73 66 23 40 44 93 85 43 63 28 98 19 29  9 34 68 95 51 55
 89 67 61  6 38 88]
[77 32 16 49 23 74 67 24 41 45 94 86 44 64 29 99 20 30 10 35 69 96 52 56
 90 68 62  7 39 89]

test

[19 74 91 58 16 20 50 55 10 35 41 87 95 72 77 80 24 89  3 46  8 53 43 64
 27 98 56 75 79  9 62  5 30  4 57 85 47 15 13 96 28 78 40 14 39 70 52 83
 17 38 86 33 65 93 44 22 21 94 49 90 18 34  1  2 63 69 12 42 61  0]
[20 75 92 59 17 21 51 56 11 36 42 88 96 73 78 81 25 90  4 47  9 54 44 65
 28 99 57 76 80 10 63  6 31  5 58 86 48 16 14 97 29 79 41 15 40 71 53 84
 18 39 87 34 66 94 45 23 22 95 50 91 19 35  2  3 64 70 13 43 62  1]

random.shuffle(list)
x트레인과 y트레인 , x test y test데이터를 섞는다
쌍으로 섞어 7:3으로 나눈다.
(70,)
(70,)
(30,)
(30,)
"""

