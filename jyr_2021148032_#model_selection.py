import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("-----------실습1-----------")
data = pd.read_csv('lin_regression_data_03.csv', names=['age', 'height'], encoding="utf-8-sig")             
print("----데이터 불러오기----")
print(data)

x = np.array(data['age'])
y = np.array(data['height'])

plt.scatter(x,y, label = 'Original Data')
plt.grid(True)
plt.xlabel("age")
plt.ylabel("height")
plt.legend()
plt.show()

print("그래프 확인")

print("-----------실습2-----------")

print("그래프 확인")
train_data = data[:20]
test_data = data[20:]

x_train_data = np.array(train_data['age'])
y_train_data = np.array(train_data['height'])


x_test_data = np.array(test_data['age'])
y_test_data = np.array(test_data['height'])

plt.scatter(x_train_data, y_train_data ,label = 'Training Data')
plt.scatter(x_test_data , y_test_data ,label = 'Test Data' , color = 'red')
plt.grid(True)
plt.xlabel("age")
plt.ylabel("height")
plt.legend()
plt.show()


print("-----------실습3-----------")

x1 = train_data["age"] #훈련집합 
y1 = train_data["height"]

x2 = test_data["age"] #테스트 집합
y2 = test_data["height"]


x1_max = max(x1) #훈련집합
x1_min = min(x1)
x1_size = x1.shape[0]

#x2_max = max(x2)  #테스트 집합
#x2_min = min(x2)
x2_size = x2.shape[0]

#mu값 계산 
def calc_mu(K, x_min , x_max):
    return [x_min + (x_max - x_min) / (K - 1) * k for k in range(K)]


#sigma값 계산
def calc_sigma(K, x_min, x_max):
    return (x_max - x_min)/(K-1)

#가우스함수 계산
def calc_gauss(x, mu, sigma):
    return np.exp(-1/2 * ((x-mu)/sigma)**2)


def save_phi(x , K ,mu ,sigma):
    result = [] #result라는 빈 리스트
    for i in range(K):
        result.append(calc_gauss(x ,mu[i],sigma)) #기저함수에 입력값 넣은 값들의 행렬만드려고
    return np.array(result).T
    #각 구한 mu값, sigma 값을 통해 가우스를 계산하여 배열에 넣는다.
    #result의 요소값은 각 특성의 데이터 값이므로 np.array로 변환하고 T시키면
    #데이터에 할당되는 특성을 뽑아올 수 있다. 


#가우시안 값을 위한 함수
def gaussian(K, x_data , x_min , x_max):
    
    mu = calc_mu(K , x_min, x_max) #mu구하기
    sigma = calc_sigma(K , x_min , x_max ) #sigma값 구하기

    #phi행렬을 구하는 함수
    phi = save_phi(x_data, K , mu , sigma)

    phi_bias = np.c_[phi, np.ones(len(x_data))] #더미값 생성
    return phi_bias


K_list = [6,7,8,9,10,11,12,13]       
   
phi_list_train = []
phi_list_test = []

#각 K값에 따른 phi값을 리스트안에 넣어 저장한다.
for i in range(len(K_list)):
    phi_list_train.append(gaussian(K_list[i], x1 , x1_min , x1_max))
    phi_list_test.append(gaussian(K_list[i], x2 , x1_min , x1_max))


#매개변수(가중치) 구하는 함수
def calc_phi_w(phi, y):
    return np.linalg.pinv(phi.T @ phi) @ phi.T @ y #train 데이터의 가중치를 구한다. 


#가중치값
w_gauss = []

for i in range(len(K_list)):
    #각각phi값에 따라 가중치를 빈 리스트에 저장(출력을 위해)
    w_gauss.append(calc_phi_w(phi_list_train[i] , y1)) 
    print("K = {} : 매개변수  = {}" .format(K_list[i] , w_gauss[i]))

    
print("-----------실습4-----------")
print("그래프 확인")

def calc_mse(y_hat , y , N):
    return sum((y_hat - y)**2)/N

mse_phi_list_train = []
mse_phi_list_test = []    

for i in range(len(K_list)):
    
    y1_predict_list_train = phi_list_train[i] @ (w_gauss[i])
    y2_predict_list_test = phi_list_test[i] @ (w_gauss[i])
    
    #mse구하는 함수를 호출해서 mse 빈 리스트에 값 저장
    mse_phi_list_train.append(calc_mse(y1_predict_list_train, y1 ,x1_size ))
    mse_phi_list_test.append(calc_mse(y2_predict_list_test, y2 , x2_size))


for i in range(len(K_list)):
    #실습6에서 K에따른 MSE값 넣은 배열 그대로 가져와서 출력
    print("K = {} / MSE = {}" .format(K_list[i] , mse_phi_list_train[i]))  
print("############훈련 MSE와 테스트 MSE 구분선###############")
for i in range(len(K_list)):
    #실습6에서 K에따른 MSE값 넣은 배열 그대로 가져와서 출력
    print("K = {} / MSE = {}" .format(K_list[i] , mse_phi_list_test[i]))
    
plt.figure()
plt.plot( K_list, mse_phi_list_train , label = "training MSE")
plt.plot( K_list, mse_phi_list_test , label = "Test MSE")
plt.grid(True)
plt.xlabel("K")
plt.ylabel("MSE")
plt.legend()
plt.show()  

print("-----------실습5-----------")
print("그래프 확인")

data = pd.read_csv('lin_regression_data_03.csv', names=['age', 'height'], encoding="utf-8-sig")             
print("----데이터 불러오기----")
print(data)
    
set_0 = np.array(data[:5])
set_1 = np.array(data[5:10])
set_2 = np.array(data[10:15])
set_3 = np.array(data[15:20])
set_4 = np.array(data[20:25])


plt.scatter( set_0.T[0]   , set_0.T[1]  ,  label = '0th_set' , color = 'blue')
plt.scatter( set_1.T[0]   , set_1.T[1]  ,  label = '1th_set' , color = 'orange')
plt.scatter( set_2.T[0]   , set_2.T[1]  ,  label = '2th_set' , color = 'green')
plt.scatter( set_3.T[0]   , set_3.T[1]  ,  label = '3th_set' , color = 'red')
plt.scatter( set_4.T[0]   , set_4.T[1]  ,  label = '4th_set' , color = 'purple')
plt.grid(True)
plt.xlabel("age[months]")
plt.ylabel("height[cm]")
plt.legend()
plt.show()


print("-----------실습6-----------")

# K=9 일때 가우스 함수를 이용한 선형 기저함수 모델 사용을 위한 변수 선언
K = 9

# 분리된데이터를 모을때 쓸 리스트 검증1개 훈련4개
train_data_list = []
test_data_list = []

w_list = [] #가중치저장할 때 쓸 리스트

#mse 저장할 때 쓰는 리스트
train_data_mse = []
test_data_mse = []

#분할되었던 데이터 모으는 리스트 
data_collect = []

#5개의 분할 데이트 다시 모으기 
for i in range(5):
    data_collect.append(data[5 * i : 5 * (i+1)])

for i in range(5):
    #5개묶음 중에 한 믂음씩 검증데이터로 변환시키기 
    s_data = np.array(data_collect[i]).reshape(-1,2).T
    t_data = np.array(data_collect[:i] + data_collect[i+1:]).reshape(-1,2).T
    
    #그래프 출력을 위해 데이터를 각각 리스트에 저장
    test_data_list.append(s_data)
    train_data_list.append(t_data)
    
    t_data_max = max(train_data_list[i][0])
    t_data_min = min(train_data_list[i][0])

    # 각 데이터의 phi값 계산하여 저장
    train_phi = gaussian(K, train_data_list[i][0], t_data_min, t_data_max)
    test_phi = gaussian(K, test_data_list[i][0], t_data_min, t_data_max)
    
    #가중치 구하기
    w_list.append(calc_phi_w(train_phi, train_data_list[i][1]))
    #주의해야할 점은 가중치 부분에서 훈련집합에 맞춘 가중치 하나만 나와야한다. 

    #phi값과 가중치 값을 사용하여 yhat 구함.
    train_y_hat = train_phi @ w_list[i]
    test_y_hat = test_phi @ w_list[i]
    
    #각 조건에 따른 mse값 구하기
    train_data_mse.append(calc_mse(train_y_hat, t_data[1] , len(train_y_hat) ))
    test_data_mse.append(calc_mse(test_y_hat, s_data[1] , len(test_y_hat)))

    
    print("K={}, 매개변수 : {}, 일반화 오차 : {}".format(i+1, w_list[i], test_data_mse[i]))
    


print("-----------실습7-----------")
print("그래프확인")

            
for i in range(5):
    plt.subplot(321+ i)
    
    #그래프를 곡선으로 만들기 위해서 np.linspace를 사용하여 데이터 늘리기 
    x_print = np.linspace(min(train_data_list[i][0]) ,  max(train_data_list[i][0]) , 100)
    #그에 따른 y예측값을 만들기 위해서 다시 가우시안함수에 넣어 구한다.
    x_print_phi = gaussian(K, x_print,  min(train_data_list[i][0]),  max(train_data_list[i][0]))
    x_print_y_hat = x_print_phi @  w_list[i]
    
    plt.plot( x_print , x_print_y_hat , 'b', label="predict, k={}-fold, MSE={}".format(i+1, round(test_data_mse[i], 5)))
    plt.plot(train_data_list[i][0], train_data_list[i][1], 'bo ', label="training set")
    plt.plot(test_data_list[i][0], test_data_list[i][1], 'o',color="orange", label="validation set")
    plt.xlabel("age[months]")
    plt.xlabel("height[cm]")
    plt.grid(True)
    plt.legend(loc="upper left")
    
plt.show()