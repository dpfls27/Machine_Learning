import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("---------------Error Back-Propagation 알고리즘 구현---------------")

def sigmoid(x): #시그모이드 함수
    return 1 / ( 1 + np.exp(-x))

def actf_grad(x):
    return (1.0-x) * x

def softmax(x): #소프트맥스 함수
    y = np.exp(x)
    sum =np.sum(y)
    return y/sum

def relu(x): # relu함수
    return np.maximum(0,x)

def relu_grad(x):   
    if x>0:
        return 1
    else:
        return 0


# Hyper parameter
learning_rate = 0.0004 #학습률
epoch = 1000 #반복 횟수

#one-hot-encoding 값으로 업데이트한 csv 파일 불러오기
data = pd.read_csv('NN_data_one_hot.csv', names=['x0', 'x1', 'x2', 'y0', 'y1', 'y2'], encoding= "utf-8-sig"  , skiprows = 1)         
       
print("-----데이터값 불러오기-----")
#print(data)

train_data, test_data = train_test_split(data, test_size=0.3)

X = np.array(train_data.iloc[:,:3]) #훈련데이터의 x0 x1 x2 총 3가지 
Y = np.array(train_data.iloc[:,3:]) #훈련데이터의 y0 y1 y2 총 3가지
X_Test = np.array(test_data.iloc[:,:3])
Y_Test = np.array(test_data.iloc[:,3:])

inputs = X.shape[1] # 훈련데이터의 특성과 동일 
hiddens_count = int(input("은닉층의 개수를 입력하세요: "))
hiddens = list(map(int, input("은닉층의 노드수를 입력하세요: ").split())) #노드 수 담은 리스트
outputs = Y.shape[1] # 클래스 수랑 출력층 수랑 동일 

""""
1. 은닉측 n개를 입력 받는다.
2. 은닉층의 갯수 만큼의 노ㅅ드 수 를 입력 받는다.
3. 가중치를 초기화 시킨다.
    - 가중치는 은닉층의 _개 이다.
    - 가중치를 _ 개를 초기화 시킨다.
4. 학습을 시작한다.
""" 

W = []

row = inputs
for col in hiddens: # [3]
    W.append(np.random.randn(row+1, col)) # W = [[4*3]]
    row = col
W.append(np.random.randn(row+1, outputs)) #[(3+1),outputs]인 애가 필요 

# 선형조합한 z 구하기 w*x 
# 활성화함수에 구한 z 집어넣기 그 값을 다시 다음 w와 내적 이 과정을 은닉층+1번 반복 
# 마지막으로 나온 z 값을 softmax같은 활성화함수에 집어 넣어서 나온 값이 최종 예측값

def forward_update(x, w_list):
    temp = np.array(x)
    f_z_list = [temp]
    for i in range(len(w_list)): # 01
        temp = np.append(temp,1) # 1 4
        z = np.dot(temp, w_list[i]) # 1 3
        f_z = activation_func_list[i](z) # 1 3
        f_z_list.append(f_z)
        temp = f_z 
        # (x+1) * W[0] -> z -> f(z)
        # (f(z)+1) * W[1] -> z -> f(z)
    return temp, f_z_list

def backward_update(y, y_pred, f_z_list):
    E_differ = 2*(y - y_pred)*(-1) # E/O1 E/O2 E/O3
    prime = activation_func_grad_list[-1](f_z_list[-1]) # O/Z4 OZ5 OZ6

    deltaW = []
    EO = []
    for i in range(len(E_differ)):
        EO.append(E_differ[i]*prime[i])
    EO = np.array(EO).reshape(-1,1) # 3, 1
    # W :             W0 W1 \
    # f_z_list     x  h1 o1
    # activation      si0 si1
    #   x w0 z si0 h1 w1 o si1 o1

    for i in range(len(W)-1, -1, -1): # 1 0
        # 마지막 가중치부터 업데이트 해서 모든 가중치를 업데이트
        f_z_list[i] #1 3 / 4 3
        a = np.r_[f_z_list[i], [1]].reshape(-1,1) # 4,1
        b = EO.reshape(1,-1)
        c = np.dot(a,b) # 4,1 1,3
        deltaW.insert(0,c)
        
        EO = np.dot(W[i], EO)[:-1]
        
        if activation_func_grad_list[i] != None:
            prime2 = activation_func_grad_list[i](f_z_list[i])

            for j in range(len(EO)):
                EO[j] =EO[j]*prime2[j]
    
    for i in range(len(deltaW)):
        W[i] = W[i] - learning_rate*deltaW[i]
        
    

activation_func_list=[sigmoid, sigmoid]
activation_func_grad_list=[None, actf_grad, actf_grad]
    
def accur(y_pred, target):
    data_index = np.argmax(y_pred)
    target_index = np.argmax(target)
    
    if data_index == target_index:
        return 1
    return 0
    

def train():
    mse_list = []
    acc_list = []
    e_list = []
    mse_list_print = []
    acc_list_print = []
    
    e_mse = 0
    e_acc = 0
    for i in range(X.shape[0]):
        y_predict, _ = forward_update(X[i], W)
        e_mse += sum((y_predict - Y[i])**2)
        e_acc +=accur(y_predict, Y[i])
    mse_list.append(e_mse / X.shape[0])
    acc_list.append(e_acc / X.shape[0])
    
    for e in range(epoch+1):    
        e_mse = 0
        e_acc = 0
        for i in range(X.shape[0]):
            y_predict, f_z_list = forward_update(X[i], W)
        
            e_mse += sum((y_predict - Y[i])**2)
            e_acc +=accur(y_predict, Y[i])
            backward_update(Y[i], y_predict, f_z_list)
 
        mse_list.append(e_mse / X.shape[0])
        acc_list.append(e_acc / X.shape[0])
        
        
        if e%100 ==0:
            e_list.append(e)
            mse_list_print.append(e_mse / X.shape[0])
            acc_list_print.append(e_acc / X.shape[0])
            print("epoch:  {}, acc: {} , mse: {} ".format(e, acc_list[-1], mse_list[-1]))
    return mse_list, acc_list, e_list , mse_list_print, acc_list_print

mse_list, acc_list, e_list, mse_list_print,acc_list_print = train()

plt.plot(e_list , mse_list_print , label = 'MSE')
plt.plot(e_list , acc_list_print , label = 'ACC')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Test

def test():
    e_mse = 0
    e_acc = 0
    for i in range(X_Test.shape[0]):
        y_predict, _ = forward_update(X_Test[i], W)
        e_mse += sum((y_predict - Y_Test[i])**2)
        e_acc +=accur(y_predict, Y_Test[i])

    print("MSE: ",e_mse / X_Test.shape[0])
    print("ACC: ",e_acc / X_Test.shape[0])
    
test()