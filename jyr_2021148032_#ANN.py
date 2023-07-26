print("---------------실습1---------------")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
def step_func(k):
    if k > 0:
        return 1
    else:
        return 0
        
X = np.array([
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,1,1]
])


y_or = np.array([0,1,1,1]) # 정답 
y_nand = np.array([1,1,1,0]) # nand게이트 정답
y_and = np.array([0,0,0,1]) #and게이트 정답
W = np.zeros(len(X[0])) 

def perceptron_fit(X,Y,epoch):
    global W
    learning_rate = 0.2
    for t in range(epoch):
        print("epoch=" , t)
        for i in range(len(X)):
            predict = step_func(np.dot(X[i],W))
            error = Y[i] - predict
            W += learning_rate*error*X[i]
            print("현재 처리 입력=", X[i], "정답=", Y[i] , "출력=", predict, "변경된 가중치=", W)
    
          
def perceptron_predict(X,Y):
    global W 
    for x in X:
        print(x[0], x[1], "=>", step_func(np.dot(x,W)))



print("                            ")
print("and 게이트 구현 후 가중치 찾기") 
perceptron_fit(X,y_and,6) # AND게이트 가중치 찾고 구현하기
perceptron_predict(X,y_and)
W_and = W

print("or 게이트 구현 후 가중치 찾기") 
W = np.zeros(len(X[0]))  
perceptron_fit(X,y_or,6) # OR게이트 가중치 찾고 구현하기
perceptron_predict(X,y_or)
W_or = W

print("                            ")
print("nand 게이트 구현 후 가중치 찾기")
W = np.zeros(len(X[0]))  
perceptron_fit(X,y_nand,6) # NAND게이트 가중치 찾고 구현하기
perceptron_predict(X,y_nand)
W_nand =W

print("                            ")
print("AND와 OR과 NAND퍼셉트론으로 XOR 게이트 구현하기")
#AND 게이트
def AND(x1, x2):
    sum = W_and[0]*x1 + W_and[1]*x2 + W_and[2]
    if sum > 0:
        return 1
    else:
        return 0
    
#NAND 게이트
def NAND(x1, x2):
    sum = W_nand[0]*x1 + W_nand[1]*x2 + W_nand[2]
    if sum > 0:
        return 1
    else:
        return 0

#OR 게이트
def OR(x1, x2):
    sum = W_or[0]*x1 + W_or[1]*x2 + W_or[2]
    if sum > 0:
        return 1
    else:
        return 0

#XOR 게이트
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


print("x = [0,0] => " , XOR(0,0))
print("x = [1,0] => " , XOR(1,0))
print("x = [0,1] => " , XOR(0,1))
print("x = [1,1] => " , XOR(1,1))


           
print("---------------실습2---------------")

def sigmoid(x): #시그모이드 함수
    return 1 / ( 1 + np.exp(-x))

def softmax(x): #소프트맥스 함수
    y = np.exp(x)
    sum =np.sum(y)
    return y/sum

def itself(x): #항등함수
    return x

def relu(x): # relu함수
    return np.maximum(0,x)


x_original = np.array([1.0,0.5])
w1 = np.array([[0.1, 0.2, 0.3], [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
w2 = np.array([[0.1, 0.2], [0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])


x = np.append(x_original, 1) # 더미값 추가
result =np.dot(x, w1) # x와 주어진 가중치 내적

#case1
result_c1 = sigmoid(result) #은닉층
result_c1 = np.append(result_c1,1) #더미값 추가
result_c1_1 = np.dot(result_c1,w2)
y_c1 = itself(result_c1_1) #출력층
print("case1의 y값", y_c1)

#case2
result_c2 = sigmoid(result) #은닉층
result_c2 = np.append(result_c2, 1)#더미값 추가
result_c2_1 = np.dot(result_c2,w2)
y_c2 = softmax(result_c2_1)#출력층
print("case2의 y값", y_c2)

#case3
result_c3 = relu(result) #은닉층
result_c3 = np.append(result_c3, 1)#더미값 추가
result_c3_1 = np.dot(result_c3,w2)
y_c3 = itself(result_c3_1)#출력층
print("case3의 y값", y_c3)

#case4
result_c4 = relu(result) #은닉층
result_c4 = np.append(result_c4, 1)#더미값 추가
result_c4_1 = np.dot(result_c4,w2)
y_c4 = softmax(result_c4_1)#출력층
print("case4의 y값", y_c4)


print("---------------실습3---------------")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
print("그래프 확인")

data = pd.read_csv('NN_data.csv', names=['x0', 'x1', 'x2', 'y'], encoding= "utf-8-sig"  , skiprows = 1)              
print("-----데이터값 불러오기-----")
print(data)

data_x = data.iloc[:,:3] #행은 전체 열은 0,1,2열만(x값만) 
data_y = data.iloc[:, 3] #행은 전체 열은 3만(y값만)
noise =np.random.normal( 0, 1.5 , size=(900,3)) # 노이즈값 랜덤

x_add_noise = data_x + noise #노이즈 추가 

set1 = x_add_noise[:300]
set2 = x_add_noise[300:600]
set3 = x_add_noise[600:]
Y = np.array(data["y"])


print("그래프 확인")
ax.scatter( np.array(set1["x0"]) , np.array(set1["x1"]) , np.array(set1["x2"]),  facecolors='none', edgecolors='blue', label='1')
ax.scatter( np.array(set2["x0"]), np.array(set2["x1"]) , np.array(set2["x2"]) ,  facecolors='none', edgecolors= 'red', label='2') 
ax.scatter( np.array(set3["x0"]), np.array(set3["x1"]) , np.array(set3["x2"]) ,  facecolors='none', edgecolors= 'orange', label='3')
ax.legend() 
plt.show()

# One-Hot Encoding 구현 
print(" ")
print("-----One-Hot Encoding 구현-----")

class_count = [] #분류할 class가 몇개인지 확인할 배열
one_index = {}

for y in Y: #data의 Y 값 1 300개 2 300개 3 300개중에서 y= 1,2,3
    if y not in class_count: # class_count배열에 y가 들어가 있지 않을때 class_count에 집어넣기(중복되는 y값 없애려고)
        class_count.append(y) # 123
        one_index[y] = len(one_index)

print("분류할 class 몇개 : " , len(class_count))

one_hot = np.zeros((len(Y), len(class_count))) # 모든 요소가 0인 배열로 초기화

for i, y in enumerate(Y): #출력층의 노드 세개로 구성
    one_hot[i, one_index[y]] = 1


print("각 class에 대해 One-Hot 표현으로 반환")
print("class 1: ", one_hot[0] , "class 2: " , one_hot[300], "class 3: " , one_hot[600])

print("NN_data.csv에 적용")
print("-----데이터값 불러오기-----")

# one_hot 배열을 one_hot.csv 파일에 저장
with open('NN_data_one_hot2.csv', 'w', newline='') as f: #data directry 안에 파일 생성 하고 싶을때 
    
    writer = csv.writer(f) #특징이름
    row_name = ["x0", "x1", "x2"] # 열 이름 다시 짓기
    
    for i in range(len(class_count)):  #원래 데이터에서는 y값이 하나였다면 one-hot값을 넣어야하니까 y0,y1,y2 총 3개를 더 만들기 위해서 써준 코드 
        row_name.append("y%d" %(i))
        
    new_update_data = np.c_[x_add_noise , np.array(one_hot)]
           
    writer.writerow(row_name) # 한줄 씩 작성
    writer.writerows(new_update_data) #여러줄 한번에 작성
    

data1 = pd.read_csv('NN_data_one_hot.csv', names=['x0', 'x1', 'x2', 'y0', 'y1', 'y2'], encoding= "utf-8-sig"  , skiprows = 1)  
print(data1)   


#2계층 신경망 구현
print(" ")
print("-----2계층 신경망 구현-----")

x_data = np.array(data1.iloc[:,:3]) #one-hot로 바꾼 후의 바뀐 csv
y_data = np.array(data1.iloc[:,3:]) 


class perceptron: #퍼셉트론 class 구현 
    def __init__(self,w):
        self.w = W
    def output(self,x):
        tmp = np.dot(self.w , np.append(1,x))
        result = 1.0*(tmp>0)
        return result
    
np.random.seed(20)     
class ANN2:
    def __init__(self , x , y ):
        self.input_size = x.shape[1] # input 속성수
        self.output_size = y.shape[1] # output 속성수
        self.hidden_layer_count = self.init_hidden_layer() # 은닉층의 수를 자유롭게 설정
        self.x = x
        self.y = y
        self.W_list = []
        self.Weight_Matrix()
        
    def init_hidden_layer(self):
        return int(input("1~10 중 은닉층의 개수를 입력하세요. "))
    
    def Weight_Matrix(self): # w 만드는 함수 
        weightSizeList = list(map(int, input("은닉층에 따른 노드 수를 입력하세요. Ex) 4 5: ").split()))
        print(weightSizeList)
        if self.hidden_layer_count != len(weightSizeList):
            print("은닉층 수와 입력 값의 개수가 다릅니다. 종료")
            exit()
        
        
        input_node = self.input_size # 맨처음 입력층의 행개수 집어넣기
        for weightSize in weightSizeList:
            self.W_list.append(np.random.randn(input_node +1 , weightSize)) # 은닉층 매개변수행렬 교재에선 (입력값(은닉층값)+1(더미값))*L의미
            input_node = weightSize # 바뀌는 행의 개수를 나타내기 위하여 
        self.W_list.append(np.random.randn(input_node+1, self.output_size)) # 출력층 매개변수 저장
    
    def fit(self, activationFunction): 
         # 빈 tmp에 바뀐 행의 값들을 업데이트 시키기 위해서
        result = []
        for i in range(self.x.shape[0]): 
            tmp = self.x[i] #입력 값
            for index in range(self.hidden_layer_count): # 출력층 부분의  제외하고 입력층부터 은닉층까지 계산
                tmp = self.active(activationFunction, tmp, index) # 활성화함수 포함해서 계산하는 함수에 집어넣고 빈 tmp에 답 저장
            # for문으로 계속 tmp에 업데이트 
            
            result.append(self.active(activationFunction, tmp, self.hidden_layer_count)) # 마지막 출력층 부분 까지 계산 
             # 이때 위에서 index값에 hidden_layer_count를 쓰는 이유는 위 for index in range(self.hidden_layer_count)에서 index는 결국 0부터 hidden_layer_count-1 인 값이니까 
             # 남은 출력층 부분의 계산 부분에서 hidden_layer_count를 씀으로써 본인의 index값 설명 가능 
        return result
        
    def active(self, activationFunction, tmp, index): # 활성화 함수 넣어서 계산하는
        x_input = np.append(tmp , 1) # 입력값에 더미값 추가
        z = np.dot(x_input, self.W_list[index]) # 선형조합
        tmp = activationFunction[i](z) # f(z) 선형조합한거 함수에 넣기
        return tmp # 함수에 넣은 값 출력 
               
activationFunction = [sigmoid, relu, softmax] # 사용할 활성화 함수 목록 

noise_x  = x_data + noise # 노이즈 추가된 x 데이터 
model =ANN2(x = noise_x, y = y_data) # 2계층 신경망에 답 넣어보기
result1 = model.fit(activationFunction) # 출력층까지 거친 최종답

pred_class = [] #y예측값 넣을 배열

for index, result in enumerate(result1):
    print(result, max(result), list(result).index(max(result))) # 순서대로 (최종답, 최종답중 가장큰 값, 그 최종답 중 가장큰 값의 index값)
    print(class_count[list(result).index(max(result))], " : ", y_data[index]) # 순서대로 (class_count(100 010 001)에서 그 최종답 중 가장큰 값의 index값을 class_count[index]로 : 원래 첫 y데이터값(one-hot변환후)))
    #큰 값(1에 가까운 값) 구하는 이유는 softmax함수 출력 사용 때문.
    pred_class.append(class_count[list(result).index(max(result))]) #출력하고하자는 y예측값 배열에 넣어주기
    


with open('NN_data_y_pred.csv', 'w', newline='') as f: #data directry 안에 파일 생성 하고 싶을때 
    
     writer = csv.writer(f) #특징이름
     row_name = ["x0", "x1", "x2" , "y" , "y_pred"] # 열 이름 다시 짓기
        
     new_update_data = np.c_[noise_x, data_y, pred_class ] # y예측값 데이터 업데이트 

           
     writer.writerow(row_name) # 한줄 씩 작성
     writer.writerows(new_update_data) #여러줄 한번에 작성
