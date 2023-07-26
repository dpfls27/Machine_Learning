import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 엑셀 데이터 불러오기
csv = pd.read_csv('linear_regression_data01.csv', names=['age', 'height'], encoding="utf-8-sig")              
print("-----데이터값 불러오기-----")
print(csv)

print("-----------실습1-----------")
print("그래프 확인")
# 각각의 점 표시하기 
x = np.array(csv['age'])
y = np.array(csv['height'])

plt.scatter(x,y, label = 'Original Data')
plt.grid(True)
plt.xlabel("age")
plt.ylabel("height")
plt.legend()
plt.show()

print("-----------실습2-----------")

#해석해 구하기
N = x.shape[0] # 데이터의 개수라고 할수 있다. 왜냐면 x도 y도 총 25개니까 x.shape구해서 N값 하나로 통일 
x_avg = np.sum(x)/N #x의 평균

w0_m = 0 #w0의 분모
w0_j = 0 #x값의 제곱의 평균을 구하기 위한 변수
w1 = 0 #바이어스

for i in range(N):
    w0_m += y[i]*(x[i]-x_avg) #해석해로 w0최적해 구하는 공식에서 분자
    w0_j += x[i]**2  #해석해로 w0최적해 구하는 공식에서 분모에 있는 x제곱의 평균을 나타내려고 만든 변수
    
w0 = (1/N)*w0_m / ((1/N)*w0_j-x_avg**2) #가중치(기울기)의 최적해 구하기

for i in range(N):  #바이어스(y절편) 구하기 
    w1 += y[i]-w0*x[i]
w1 = w1/N  
  
print("w0 = %0.8f , w1 = %0.8f" %(w0,w1)) 

print("-----------실습3-----------")
print("그래프 확인")
y_regression = w0*x+w1

plt.scatter(x,y, label = 'Original Data')
plt.plot( x, w0*x+ w1, label = 'Linear Regression', color = 'red')
plt.grid(True)
plt.xlabel("age")
plt.ylabel("height")
plt.legend()
plt.show()

print("-----------실습4-----------")

#평균제곱오차 
MSE = 0 #평균제곱오차 변수 정의

for i in range(N): #MSE의 공식 바로 대입
    MSE += (w0*x[i]+w1-y[i])**2
MSE = MSE/N

print("MSE = %0.8f" %MSE)

print("-----------실습5-----------")

#경사하강법
learning_rate = 0.0085 #학습률 임의 설정
epoch = 3000 #반복횟수(4주차 과제 결괏값 그래프 참고하여 3000으로 고정시킴)

np.random.seed(20)
GD_W = np.random.randn(2) #정규 분포 따르는 -1~1 사이 값으로 w0과 w1의 초기값 저장

init_W = [0 for i in range(2)] # GD_W의 완전 초기값을 따로 저장해놓고 나중에 출력하기 위해서 빈 리스트를 만들어주고, 값을 저장시킨다.
init_W[0] = GD_W[0]
init_W[1] = GD_W[1]

MSE_list = [] #경사하강법의 반복 횟수에 따른 MSE와 매개변수를 그래프로 나타내기 위해, 빈 리스트를 만들어 반복횟수만큼의 개수의 값들을 저장시킨다. 
w = [init_W[0]]
b = [init_W[1]]

y_predict = x*GD_W[0]+ GD_W[1]
MSE_list.append(((y_predict-y)**2).mean())

x_line = [0] # 선별한 데이터를 출력하기 위한 x축 좌표

for e in range(1,epoch+1):
    GD_W[0] = GD_W[0] - learning_rate*2*((y_predict-y)*x).mean() #경사하강법 업데이트 규칙에 따라 계산(단, 가중치와 바이어스의 규칙이 다르기 때문에 나눔) 
    GD_W[1] = GD_W[1] - learning_rate*2*(y_predict-y).mean()
    
    y_predict = x*GD_W[0]+ GD_W[1] #y의 예측값
    gd_MSE = ((y_predict-y)**2).mean() #MSE값
    
    
    if e % 100 ==0: # 경사하강법 업데이트를 100번 할때마다 w0, w1, mse의 값을 출력.
        x_line.append(e)
        MSE_list.append(gd_MSE) 
        w.append(GD_W[0])      
        b.append(GD_W[1])  
        print("epoch : %d =========> w0: %0.8f w1: %0.8f gd_MSE: %0.8f" %(e, GD_W[0], GD_W[1], gd_MSE))
       

print("-----------실습6-----------")

print("학습율: {} , 반복 횟수: {} , 초기값:[ {:.8f} , {:.8f}] , 최종 평균제곱오차: [{:.8f}] , 최적 매개변수: {}".format(learning_rate , epoch, *init_W, gd_MSE, GD_W))

print("-----------실습7-----------")

print("그래프확인")
#경사하강법에 따른 MSE 그래프 

plt.plot(MSE_list)
plt.grid(True)
plt.xlabel('STEP')
plt.ylabel('MSE')
plt.show() 


#경사하강법에 따른 매개변수 값 그래프 
plt.plot( x_line , b, 'r', label = 'GD_W[1]')  
plt.plot( x_line , w, 'b', label = 'GD_W[0]')
plt.grid(True)
plt.xlabel('STEP')
plt.ylabel('GD_W[0] & GD_W[1]')
plt.legend()
plt.show() 

print("-----------실습8-----------")
print("그래프확인")
gd_model = GD_W[0]*x + GD_W[1]
plt.scatter(x,y, label = 'Original Data')
plt.plot( x , w0*x+ w1, label = 'Linear Regression', color = 'red')
plt.plot(x, gd_model ,'g', label = 'Linear Regression-GD', )
plt.grid(True)
plt.xlabel("age")
plt.ylabel("height")
plt.legend()
plt.show()