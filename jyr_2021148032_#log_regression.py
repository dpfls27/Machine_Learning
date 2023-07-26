import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("---------------실습1---------------")

print("----------문제1----------")
print("그래프 확인")

def sigmoid(z): #시그모이드함수 선언
    return 1/(1 + np.exp(-z))

z = np.arange(-10, 10, 0.1)

plt.plot(z , sigmoid(z))
plt.grid(True)
plt.show()

print("----------문제2----------")
print("그래프 확인")

# 엑셀 데이터 불러오기
data = pd.read_csv('binary_data_insect.csv', names=['weight', 'gender'], encoding="utf-8-sig" , skiprows = 1)              
print("-----데이터값 불러오기-----")
print(data)

x = np.array(data['weight'])
y = np.array(data['gender'])

plt.plot(x[:5],y[:5], 'o' ,label = 'female')
plt.plot(x[5:],y[5:], '^',label = 'male')
plt.xlabel("weight")
plt.ylabel("gender")
plt.legend()
plt.grid(True)
plt.show()


print("----------문제3----------")

#경사하강법
learning_rate  = 0.0005 #학습률
epoch = 240000 #반복 횟수
N = x.shape[0]

np.random.seed(40)
GD_W = np.random.randn(2)  #정규 분포 따르는 -1~1 사이 값으로 w0과 w1의 초기값 저장

init_W = [0 for i in range(2)] # GD_W의 완전 초기값을 따로 저장해놓고 나중에 출력하기 위해서 빈 리스트를 만들어주고, 값을 저장시킨다.
init_W[0] = GD_W[0]
init_W[1] = GD_W[1]

w = [init_W[0]] # 가중치 담을 배열
b = [init_W[1]] # 바이어스 담을 배열
cee_list = [] #cee 담을 배열 
print_e = [0] #간단한 그래프 출력을 위한 배열

z = GD_W[0]*x + GD_W[1]
p = sigmoid(z) # sigmoid함수 넣었을때의 값 저장
cee_list.append((-1)*(y*np.log(p) + (1-y)*np.log(1-p)).mean()) #cee값 저장 

for e in range(1,epoch+1):
    GD_W[0] = GD_W[0] - learning_rate*((p-y)*x).mean() #경사하강법 업데이트 규칙에 따라 계산(단, 가중치와 바이어스의 규칙이 다르기 때문에 나눔) 
    GD_W[1] = GD_W[1] - learning_rate*(p-y).mean()
    
    z = GD_W[0]*x + GD_W[1]
    p = sigmoid(z)
    cee = (-1)*(y*np.log(p) + (1-y)*np.log(1-p)).mean()
    
    if e % 10000 ==0: # 경사하강법 업데이트를 10000번에 한번씩 출력하기 위해 써줌.
         print_e.append(e)
         w.append(GD_W[0])      
         b.append(GD_W[1])
         cee_list.append(cee)
         print("epoch : %d =========> w0: %0.8f w1: %0.8f cee: %f" %(e, GD_W[0], GD_W[1] ,cee))


print("GD종료")
print("w0 = %0.8f w1 = %0.8f" %(GD_W[0] , GD_W[1]))

plt.plot(print_e, w)
plt.plot(print_e, b)
plt.xlabel("epoch")
plt.ylabel("w")
plt.grid(True)
plt.show()

plt.plot(print_e, cee_list)
plt.xlabel("epoch")
plt.ylabel("CEE")
plt.grid(True)
plt.show()

print("----------문제4----------")

z = GD_W[0]*x + GD_W[1] # 경사하강법을 통해 구한 가중치로 정해진 z
gd_p = sigmoid(z) # 확률 변환값을 구한다. 
count = []

for i in range(N):
    if gd_p[i] > 0.5:
        y_predict = 1
        if y_predict== y[i]:
            count.append(i)
            
    else:
        y_predict = 0
        if y_predict== y[i]:
            count.append(i)

accuracy = (len(count)/10)*100

print("훈련결과, 정확도: %0.1f%%" %accuracy)
print("w0 = %0.8f w1 = %0.8f" %(GD_W[0] , GD_W[1]))

k = [ 22, 25, 84, 71, 46, 33, 54, 37, 61 , 59]
count_predict = []
print("predict 함수 판별")

def predict(k):
    z = GD_W[0]*k + GD_W[1]
    gd_p_predict = sigmoid(z)
    return gd_p_predict

for i in range(10):
    pred_p = predict(k[i])
    
    if pred_p > 0.5:
        count_predict.append(1)
    else:
        count_predict.append(0)


print("임의의 데이터 10개: " ,k)   
print("임의의 데이터에 대한 0또는 1 예측 판별: ", count_predict)

plt.plot(x , y ,'o')
plt.plot(x , z)
#plt.plot(k , count_predict , '^') #임의의 10개 데이터에 대한 예측 그래프 그리기
plt.ylim([-0.2 , 1.2])
plt.xlabel("weight")
plt.ylabel("gender")
plt.grid(True)
plt.show()


print("---------------실습2---------------")
print("----------문제1----------")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
print("그래프 확인")

# 엑셀 데이터 불러오기
data_Iris = pd.read_csv('iris.csv', names=['sepal_length','petal_length' ,'variety'], encoding="utf-8-sig" , skiprows = 1)              
print("-----데이터값 불러오기-----")
print(data_Iris)
data_Iris['variety'].replace({'Setosa': 0, 'Versicolor': 1}, inplace=True) # Setosa를 0 , Versicolor를 1로 변환하는 작업

x0 = np.array(data_Iris['sepal_length'])
x1 = np.array(data_Iris['petal_length'])
y = np.array(data_Iris['variety'])

ax.scatter(x0, x1 ,y)
ax.grid(True)
ax.set_xlabel("sepal_length")
ax.set_ylabel("petal_length")
ax.set_zlabel("variety")       
plt.show()

print("----------문제2----------")

#경사하강법
learning_rate  = 0.0005 #학습률
epoch = 20000 #반복 횟수
N = x0.shape[0]

#-3~3중에서 랜덤값 뽑기 -> 가중치의 초기값 의미
np.random.seed(3)
GD_W = np.random.uniform(-3,3, size = 3)

init_W = [0 for i in range(3)] # GD_W의 완전 초기값을 따로 저장해놓고 나중에 출력하기 위해서 빈 리스트를 만들어주고, 값을 저장시킨다.
init_W[0] = GD_W[0]
init_W[1] = GD_W[1]
init_W[2] = GD_W[2] #얘는 바이어스 초기값

w0 = [init_W[0]]
w1 = [init_W[1]]
b = [init_W[2]]

cee_list_1 = [] #구한 cee 값 담을 배열
print_e_1 = [0] #간단한 그래프 출력을 위한 배열

z = GD_W[0]*x0 + GD_W[1]*x1+ GD_W[2]
p = sigmoid(z) # sigmoid함수 넣었을때의 값 저장
cee_list_1.append((-1)*(y*np.log(p) + (1-y)*np.log(1-p)).mean()) #cee값 저장 

#경사하강법 시작
for e in range(1,epoch+1):
    GD_W[0] = GD_W[0] - learning_rate*((p-y)*x0).mean() #경사하강법 업데이트 규칙에 따라 계산(단, 가중치와 바이어스의 규칙이 다르기 때문에 나눔) 
    GD_W[1] = GD_W[1] - learning_rate*((p-y)*x1).mean()
    GD_W[2] = GD_W[2] - learning_rate*(p-y).mean()
    
    z = GD_W[0]*x0 + GD_W[1]*x1+ GD_W[2]
    p = sigmoid(z)
    cee = (-1)*(y*np.log(p) + (1-y)*np.log(1-p)).mean()
    
    if e % 1000 ==0: # 경사하강법 업데이트를 1000번에 한번씩 출력하기 위해 써줌.
        print_e_1.append(e)
        w0.append(GD_W[0])      
        w1.append(GD_W[1])
        b.append(GD_W[2])
        cee_list_1.append(cee)
        print("epoch : %d =========> w0: %0.8f w1: %0.8f w2: %0.8f cee: %f" %(e, GD_W[0], GD_W[1] , GD_W[2] ,cee))
        
print("GD종료")
print("w0 = %0.8f w1 = %0.8f w2: %0.8f " %(GD_W[0] , GD_W[1], GD_W[2]))

plt.plot(print_e_1, w0)
plt.plot(print_e_1, w1)
plt.plot(print_e_1, b)
plt.xlabel("epoch")
plt.ylabel("w")
plt.grid(True)
plt.show()

plt.plot(print_e_1, cee_list_1)
plt.xlabel("epoch")
plt.ylabel("CEE")
plt.grid(True)
plt.show()


print("----------문제3----------")

z = GD_W[0]*x0 + GD_W[1]*x1+ GD_W[2] # 경사하강법을 통해 구한 가중치로 정해진 z
gd_p_1 = sigmoid(z) # 확률 변환값을 구한다. 
count_1 = []

for i in range(N): #확률 변환 후 p>0.5 , p<0.5를통해 예측모델 구하기 
    if gd_p_1[i] > 0.5: 
        y_predict = 1
        if y_predict== y[i]:
            count_1.append(i)
            
    else:
        y_predict = 0
        if y_predict== y[i]:
            count_1.append(i)

accuracy = (len(count_1)/100)*100 # 비율구하기 

print("훈련결과, 정확도: %0.1f%%" %accuracy)
print("w0 = %0.8f w1 = %0.8f w2: %0.8f" %(GD_W[0] , GD_W[1] , GD_W[2]))


# Decision boundary 그래프 그리기 
sepal_length_data = np.linspace(4.0 , 7.0, 1000) # 각자리의 시작점에서 끝점까지 1000개의 데이터 생성
petal_length_data = np.linspace(0 , 5 ,1000)

SL , PL = np.meshgrid(sepal_length_data, petal_length_data) #1차원에서 2차원으로 확장한다는 의미
z = GD_W[0]*SL + GD_W[1]*PL+ GD_W[2] #확장된 차원에 맞는 decision boundary

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(x0, x1 ,y)
ax.plot_surface(SL, PL , z , cmap = 'plasma')
ax.set_xlabel('sepal')
ax.set_ylabel("petal")
ax.set_zlabel("variety" , rotation = 0)   
ax.set_title("Emperical Solution")    
plt.show()
 
k1 = [ 3.2, 4.54 , 5 , 3.62, 7.1, 5.2, 4.7, 3.8, 3.28, 4.45] #sepal 임의의 데이터 10개
k2 = [ 1.2, 1.12, 1.4, 5.78, 2.09, 1.7, 2.58, 4.78, 3.05, 8.24] #petal 임의의 데이터 10개
count_predict_iris = [] # 예측 판별 0과 1을 담을 배열
print("predict 함수 판별")

def predict(x, x0):
    z = GD_W[0]*x + GD_W[1]*x0+ GD_W[2]
    iris_p_predict = sigmoid(z)
    return iris_p_predict

for i in range(10):
    pred_p_iris = predict(k1[i], k2[i])
    
    if pred_p_iris > 0.5:
        count_predict_iris.append(1)
    else:
        count_predict_iris.append(0)
        
print("임의의 데이터 10개 sepal: " , k1 , "petal: ", k2)   
print("임의의 데이터에 대한 0또는 1 예측 판별: ", count_predict_iris)