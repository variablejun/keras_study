weight = 0.4
input = 0.4

goal_prediction = 0.8

lr = 0.25
#러닝레이크 조절이 중요 ,러닝레이트의 원리
epoch = 20

for iter in range(epoch):
     pred = input * weight
     error = (pred - goal_prediction) **2

     print('error:' + str(error) + '\t Prediction:' + str(pred))

     up_pred = input * (weight  + lr)
     up_error = (goal_prediction - up_pred ) **2

     down_pred = input * (weight  - lr)
     down_error = (goal_prediction - down_pred ) **2

     if(down_error < up_error):
          weight = weight - lr
     if(down_error > up_error):
          weight = weight + lr    

'''
error:0.4096     Prediction:0.16000000000000003
error:0.3600000000000001         Prediction:0.2
error:0.31360000000000005        Prediction:0.24
error:0.27040000000000003        Prediction:0.27999999999999997
error:0.23040000000000005        Prediction:0.32
error:0.19360000000000005        Prediction:0.36
error:0.16000000000000006        Prediction:0.39999999999999997
error:0.12960000000000008        Prediction:0.43999999999999995
error:0.10240000000000005        Prediction:0.48
error:0.07840000000000001        Prediction:0.52
error:0.0576     Prediction:0.56
error:0.03999999999999998        Prediction:0.6000000000000001
error:0.025599999999999973       Prediction:0.6400000000000001
error:0.014399999999999972       Prediction:0.6800000000000002
error:0.006399999999999976       Prediction:0.7200000000000002
error:0.0015999999999999851      Prediction:0.7600000000000002
error:4.930380657631324e-32      Prediction:0.8000000000000003
error:0.0015999999999999851      Prediction:0.7600000000000002
error:4.930380657631324e-32      Prediction:0.8000000000000003
error:0.0015999999999999851      Prediction:0.7600000000000002
'''