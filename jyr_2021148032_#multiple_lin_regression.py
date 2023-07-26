import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

csv = pd.read_csv('multiple_linear_regression_data.csv', names=['height', 'weight' ,'label' ], encoding="utf-8-sig" , skiprows = 1) #문자인 1행 빼고 80개 불러오기              
print("-----데이터값 불러오기-----")
print(csv)

print("-----------실습1-----------")
print("그래프 확인")
# 각각의 점 표시하기 
height = np.array(csv['height'])
weight = np.array(csv['weight'])
label = np.array(csv['label'])

ax.scatter(height, weight , label)
ax.grid(True)
ax.set_xlabel("height")
ax.set_ylabel("weight")
ax.set_zlabel("age" , rotation = 0)       
plt.show()

csv_size = csv.shape[0]
x = np.c_[height ,weight] #입력값인 x끼리 모아주기
y = label

print("-----------실습2-----------")
print("그래프 확인")
x_bias = np.c_[x , np.ones(csv_size)] # height, weight,1 를 의미
y = y.reshape((csv_size , 1))

analytic_W = np.linalg.pinv(x_bias.T @ x_bias) @ x_bias.T @ y #최적 가중치와 바이어스 값을 구하는 법
print("해석해 =  " , analytic_W)

height_data = np.linspace(55,190,1000) # 각자리의 시작점에서 끝점까지 1000개의 데이터 생성
weight_data = np.linspace(10,100,1000)

Height , Weight = np.meshgrid(height_data, weight_data) #1차원에서 2차원으로 확장한다는 의미
analytic_y = analytic_W[0]*Height + analytic_W[1]*Weight + analytic_W[2] # 해석해로 구한 값의 평면 방정식

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(height ,weight, label)
ax.plot_surface(Height, Weight , analytic_y , cmap = 'plasma')
ax.set_xlabel('height')
ax.set_ylabel("weight")
ax.set_zlabel("age" , rotation = 0)   
ax.set_title("Analytic Solution")    
plt.show()

print("-----------실습3-----------")

MSE = 0 #평균제곱오차의 변수 
y_predict_eq = analytic_W[0]*height + analytic_W[1]*weight + analytic_W[2]
y_predict_martrix = x_bias @ analytic_W

MSE = sum((y_predict_martrix-y)**2)
MSE = MSE/csv_size
 
print( "평균제곱오차 = " , MSE)

print("-----------실습4-----------")
#경사하강법
learning_rate  = 0.000055 #학습률
epoch = 1200000 #반복 횟수

np.random.seed(50)
GD_W = np.random.rand(3,1) #랜덤값 추출

init_W = [0 for i in range(3)] # GD_W의 완전 초기값을 따로 저장해놓고 나중에 출력하기 위해서 빈 리스트를 만들어주고, 값을 저장시킨다.
init_W[0] = GD_W[0][0]
init_W[1] = GD_W[1][0]
init_W[2] = GD_W[2][0] #얘는 바이어스 초기값

MSE_list = [] #경사하강법의 반복 횟수에 따른 MSE와 매개변수를 그래프로 나타내기 위해, 빈 리스트를 만들어 반복횟수만큼의 개수의 값들을 저장시킨다. 
w0 = [init_W[0]] #초기값 미리 넣어주기( 안넣어주면 경사하강법 이후 바뀐 초기값들만 들어 가기 때문에 )
w1 = [init_W[1]]
b= [init_W[2]]  #얘는 바이어스 초기값


y_predict_gd = x_bias @ GD_W

MSE_list.append(sum((y_predict_gd-y)**2)/csv_size)

for e in range(1, epoch+1):
    GD_W = GD_W - learning_rate * 2 / csv_size * x_bias.T @ (y_predict_gd-y)
    
    # GD_W[0][0] = GD_W[0][0] - learning_rate*2/csv_size*(height.T @ (y_predict_gd-y))
    # GD_W[1][0] = GD_W[1][0] - learning_rate*2/csv_size*((y_predict_gd-y).T @ weight)
    # GD_W[2][0] = GD_W[2][0] - learning_rate*2/csv_size*sum(y_predict_gd-y)
    
    y_predict_gd = x_bias @ GD_W
    gd_MSE = sum((y_predict_gd-y)**2)/csv_size

    
    if e % 100 ==0: # 경사하강법 업데이트를 100번 할때마다 w0, w1,,w2 mse의 값을 출력.
        MSE_list.append(gd_MSE) 
        w0.append(GD_W[0])      
        w1.append(GD_W[1])
        b.append(GD_W[2])  
        print("epoch : %d =========> w0: %0.8f w1: %0.8f w2: %0.8f gd_MSE: %0.8f" %(e, GD_W[0], GD_W[1], GD_W[2], gd_MSE))
       

height_data = np.linspace(55,190,1000) # 각자리의 시작점에서 끝점까지 1000개의 데이터 생성
weight_data = np.linspace(10,100,1000)

Height , Weight = np.meshgrid(height_data, weight_data) #1차원에서 2차원으로 확장한다는 의미
y_predict_gd_gh = GD_W[0]*Height + GD_W[1]*Weight + GD_W[2]


#경사하강법 그래프 
fig = plt.figure()
ax = fig.add_subplot(122,projection = '3d')
ax.scatter(height ,weight, label)
ax.plot_surface(Height, Weight , y_predict_gd_gh , cmap = 'plasma')
ax.set_xlabel('height')
ax.set_ylabel("weight")
ax.set_zlabel("age" , rotation = 0)   
ax.set_title("Gradient Decent Method")    

#해석해 그래프
ax = fig.add_subplot(121, projection = '3d')
ax.scatter(height ,weight, label)
ax.plot_surface(Height, Weight , analytic_y , cmap = 'plasma')
ax.set_xlabel('height')
ax.set_ylabel("weight")
ax.set_zlabel("age" , rotation = 0)   
ax.set_title("Analytic Solution")    
#plt.show()


print("-----------실습5-----------")
#가우스 함수를 이용한 선형 기저함수 회귀모델의 최적 해석해

csv1 = pd.read_csv('linear_regression_data01.csv', names=['age', 'height'], encoding="utf-8-sig")              
print("-----지난 데이터값 불러오기-----")
print(csv1)

x1 = csv1["age"]
y1 = csv1["height"]

x1_max = max(x1)
x1_min = min(x1)
x1_size = x1.shape[0]

#mu값 계산 
def calc_mu(K):
    return [x1_min + (x1_max - x1_min) / (K - 1) * k for k in range(K)]

#sigma값 계산
def calc_sigma(K):
    return (x1_max - x1_min)/(K-1)

#가우스함수 계산
def calc_gauss(x, mu, sigma):
    return np.exp(-1/2 * ((x-mu)/sigma)**2)

#각 구한 mu값과 가우스를 계산하여 배열에 넣는다.
#result의 요소값은 각 특성의 데이터 값이므로 np.array로 변환하고 T시키면
#데이터에 할당되는 특성을 뽑아올 수 있다. 
def save_phi(x,K,mu,sigma):
    result = [] #result라는 빈 리스트
    for i in range(K):
        result.append(calc_gauss(x ,mu[i],sigma)) #기저함수에 입력값 넣은 값들의 행렬만드려고
    return np.array(result).T


#가우시안 값을 위한 함수
def gaussian(K, x_data):
    
    mu = calc_mu(K) #mu구하기
    sigma = calc_sigma(K) #sigma값 구하기
    
    #기본데이터값의 손상을 방지하기 위해 복사로 데이터 먼저 가져오기
    x_line = x_data.copy()
    #print("K = {}: mu = {} , sigma = {}".format(K,mu,sigma))
    
    #phi행렬을 구하는 함수
    phi = save_phi(x_line , K , mu , sigma)
    
    #y_predict = phi0(x) + phi1(x) + phi2(x) + bias
    phi_bias = np.c_[phi , np.ones(x1_size)] #더미값 생성
    return phi_bias


def calc_mse(y_hat , y, N):
    return sum((y_hat - y)**2)/N #행렬의 각 행 데이터 차의 제곱을 각각 구하고 모두 더해서 데이터 갯수로 눔
   
    
K_list = [3,5,8,10]
phi_list = []

#각 K값에 따른 phi값을 리스트안에 넣어 저장한다.
for i in range(len(K_list)):
    phi_list.append(gaussian(K_list[i],x1))


#매개변수(가중치) 구하는 함수
def calc_phi_w(phi):
    return np.linalg.pinv(phi.T @ phi) @ phi.T @ y1 #(4*25 @ 25*4)^T @ 4*25 @ 25*1 

#가중치값
w_gauss = []
for i in range(len(K_list)):
    #각각phi값에 따라 가중치를 빈 리스트에 저장(출력을 위해)
    w_gauss.append(calc_phi_w(phi_list[i])) 
    print("K = {} : 매개변수  = {}" .format(K_list[i] , w_gauss[i]))
    

print("-----------실습6-----------")

y1_predict_list = []
mse_phi_list = []

for i in range(len(K_list)):
    
    y1_predict_list.append(phi_list[i] @ w_gauss[i])
    # 25*4 @ 4*1 => 25*1의 예측값을 빈 리스트에 넣어준다
    mse_phi_list.append(calc_mse(y1_predict_list[i], y1 ,x1_size ))
    
    #mse구하는 함수를 호출해서 mse 빈 리스트에 값 저장
    

result_datamap = np.array(x1) #먼저 고정될 x1의 값을 빈 리스트에 넣어줌
  
for i in range(len(K_list)):
    result_datamap = np.c_[result_datamap , y1_predict_list[i]]
    # result_datamap 은 x의 값과 y예측값이 같이 들어있는 배열로 
    # K=3이라고 하면 x의 값 25개가 한 열로 나머지 K+1개의 yhat0 , yhat1, yhat2, yhat3이 각각 한열로 총 25행 5열의 행렬완성.
    
#이때 x값이 무작위 순이라서 그래프를 그리면 y값이 무작위 적으로 나오기 때문에 x의 값을 
#순서대로 sort해줘야함.

sorted_result_datamap = sorted(result_datamap, key=lambda x: x[0])
print(y1_predict_list)
#여기서 K=3이라고 할때 sorted_result_datamap를 transpose하여 
#이 배열의 각 행이 순서대로 x값25개 / yhat0 25개 / yhat1 25개/ yhat2 25개 / yhat3 25개가 되도록 한다. 
#print(sorted_result_datamap)
#print("#################################")
#print(np.array(sorted_result_datamap))
#print("#################################")
sorted_result_datamap = np.array(sorted_result_datamap).T
#print(np.array(sorted_result_datamap).T)

#그래프 출력 
plt.figure(figsize = (30,30))
for i in range(4): #K = 3,5,8,10인거 출력해야 해서 for문으로 출력값 한번만 쓰려고
    plt.subplot(2,2,1 + i)
    plt.scatter(x1,y1,label = "original") #점으로 나타낸 본 데이터
    # 비선형 그래프
    plt.plot(sorted_result_datamap[0], sorted_result_datamap[i+1] , 'r', label = "predict, K = {} " .format(K_list[i]))
    plt.xlabel("age")
    plt.ylabel("height")
    plt.title("Regression with gaussian basis function")
    plt.grid(True)
    plt.legend()
plt.show()

print("-----------실습7-----------")

#각 K개에 대한 평균 제곱오차 구하기 
for i in range(len(K_list)):
    #실습6에서 K에따른 MSE값 넣은 배열 그대로 가져와서 출력
    print("K = {} / MSE = {}" .format(K_list[i] , mse_phi_list[i]))

plt.figure()
plt.stem( K_list, mse_phi_list , label = "MSE")
plt.grid(True)
plt.xlabel("K")
plt.ylabel("MSE")
plt.legend()
plt.show()  
