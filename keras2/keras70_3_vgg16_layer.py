from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))
vgg16.trainable=False 
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
# vgg16을 연산하지않아 가중치에 변화없이 기존에 모델링으로 주는것, 가중치 동결 훈련 동결

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))
'''

'''
import pandas as pd
pd.set_option('max_colwidth',-1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
result = pd.DataFrame(layers, columns=['Layer Type','Layer Name','Layer Trainable'])

print(result)
'''
                                                                            Layer Type Layer Name  Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x000001F67FBF1B80>  vgg16      False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x000001F60B525CD0>           flatten    True
2  <tensorflow.python.keras.layers.core.Dense object at 0x000001F60B51ACA0>             dense      True
3  <tensorflow.python.keras.layers.core.Dense object at 0x000001F60B51AC40>             dense_1    True
'''