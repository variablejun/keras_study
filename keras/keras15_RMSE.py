import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3,4,5]) #numpy 형식로바꿔줌 완성후 출력스샷(loss, predic)
y = np.array([1,2,4,3,5]) # 판단은 loss값으로

x_pred = np.array([6])

model = Sequential()
model.add(Dense(1, input_dim = 1))

model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=1000, batch_size=1)

loss = model.evaluate(x,y)
print('Loss : ', loss)

y_predic = model.predict(x_test)
print('예측값 : ', y_pred)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y, x_pred) # y의원래값과 y의 예측값

print('r2 score : ', r2)
def rems(y_test, y_predic): # 함수로 평가지표를 만든다
    return np.sqrt(mean_squared_error(y_test, y_predic))
REMS = rems(y_test, y_predic)

# 리더보드 방식 -> 최고점을 항상 갱신시키는 방식
# 0.9까지