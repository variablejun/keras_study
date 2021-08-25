import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input,Conv2D

(x_train, y_train),(x_test,y_test)= mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255
x_test= x_test.reshape(10000,28*28).astype('float32')/255

def build_model(drop=0.5,optimizer='adam'):
     inputs= Input(shape=(28*28), name='Input')
     x = Dense(512, activation='relu',name='hidden1')(inputs)
     x = Dropout(drop)(x)
     x = Dense(256, activation='relu',name='hidden2')(x)
     x = Dropout(drop)(x)
     x = Dense(128, activation='relu',name='hidden3')(x)
     x = Dropout(drop)(x)
     outputs = Dense(10,activation='softmax',name='outputs')(x)
     model = Model(inputs=inputs,outputs =outputs )
     model.compile(optimizer=optimizer,metrics=['acc'],loss='categorical_crossentropy')
     return model

def create_hyperparameter():
     batches = [1000,2000,3000,4000,5000]
     optimizers = ['rmsprop','adam','adadelta'] 
     dropout = [0.5,0.6,0.7]
     return {'batch_size':batches, 'optimizer': optimizers, 'drop':dropout}

hyperparameters = create_hyperparameter()
print(hyperparameters)
#{'batch_size': [10, 20, 30, 40, 50], 'optimizer': ['rmsprop', 'adam', 'adadelta'], 'drop': [0.1, 0.2, 0.3]}
#model2 = build_model()
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier# 텐서모델을 사이킷런에서 돌릴수있도록하는것, 텐서를 사이킷런 형태로 래핑
model2 = KerasClassifier(build_fn=build_model,verbose=1)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
model = RandomizedSearchCV(model2, hyperparameters,cv=5)
model.fit(x_train,y_train,verbose=1,epochs=3, validation_split=0.2)

print(model.best_estimator_)
print(model.best_params_)
print(model.best_score_)
acc = model.score(x_test,y_test)
print(acc)
'''
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001BCCE273100>
{'optimizer': 'rmsprop', 'drop': 0.5, 'batch_size': 1000}
0.9427833318710327
10/10 [==============================] - 0s 3ms/step - loss: 0.1547 - acc: 0.9530
0.953000009059906
'''