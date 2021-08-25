import numpy as np
'''
cnn 으로변경
파라미터 변경해보기 에포는 123
노드수, 엑티베이션,러닝레이트 설정
나중에 레이어도 파라미터로 구성
'''
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Input,Conv2D,Flatten

(x_train, y_train),(x_test,y_test)= mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#x_train = x_train.reshape(60000,28*28).astype('float32')/255
#x_test= x_test.reshape(10000,28*28).astype('float32')/255

def build_model(drop=0.5,optimizer='adam',units=32,activation='relu'):
     inputs= Input(shape=(28,28,1), name='Input')
     x = Conv2D(units,kernel_size=(2,2), activation='relu',name='hidden1')(inputs)
     x = Dropout(drop)(x)
     x = Conv2D(units,kernel_size=(2,2), activation='relu',name='hidden2')(x)
     x = Dropout(drop)(x)
     x = Conv2D(units,kernel_size=(2,2), activation='relu',name='hidden3')(x)
     x = Dropout(drop)(x)
     x = Flatten()(x)
     outputs = Dense(10,activation='softmax',name='outputs')(x)
     model = Model(inputs=inputs,outputs =outputs)
     model.compile(optimizer=optimizer,metrics=['acc'],loss='categorical_crossentropy')

     return model

def create_hyperparameter():
     batches = [1000,2000,3000,4000,5000]
     optimizers = ['rmsprop','adam','adadelta'] 
     dropout = [0.2,0.3,0.4]
     activations = ['relu','sigmoid']
     filterz = [16,32]
     learning_rate = [0.1,0.001,0.0001]
     return {'batch_size':batches, 'optimizer': optimizers, 'drop':dropout,
     'activation':activations,'units':filterz}

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


<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000018DC1E49C40>
{'units': 16, 'optimizer': 'adam', 'drop': 0.5, 'batch_size': 3000, 'activation': 'relu'}
0.8399000048637391
4/4 [==============================] - 0s 13ms/step - loss: 0.7129 - acc: 0.8507
0.8507000207901001


<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000017BDA340A60>
{'units': 32, 'optimizer': 'adam', 'drop': 0.3, 'batch_size': 1000, 'activation': 'sigmoid'}
0.9024500012397766
10/10 [==============================] - 0s 9ms/step - loss: 0.2537 - acc: 0.9361
0.9361000061035156
'''