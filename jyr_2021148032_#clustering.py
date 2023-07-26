import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

print("----------실습1----------")

df = pd.read_csv('Mall_Customers.csv') 
#print(df.head(5))

data = df[['Annual Income (k$)', 'Spending Score (1-100)']]
k = 3
km = KMeans(n_clusters= k , random_state= 10 , n_init = 10)

df['cluster'] = km.fit_predict(data)

final_centroid = km.cluster_centers_
#print(final_centroid)

plt.figure(figsize= (8,8))
for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'Annual Income (k$)'], df.loc[df['cluster'] == i, 'Spending Score (1-100)'], label = 'cluster' + str(i))
    
plt.scatter(final_centroid[:,0], final_centroid[:,1], s=50, c = 'violet', marker = 'x', label = 'Centroids')
plt.legend()
plt.title(f'K = {k} results', size = 15)
plt.xlabel('Annual Income', size = 12)
plt.ylabel('Spending Score', size = 12)
plt.show()

def elbow(X):
    sse = []
    for i in range(1,11):
        km = KMeans(n_clusters= i, random_state= 0)
        km.fit(X)
        sse.append(km.inertia_)
        
    plt.plot(range(1,11), sse, marker = 'o')
    plt.xlabel('# of clusters')
    plt.ylabel('Interia')
    plt.show()

elbow(data)



#print(df)

print("----------실습2----------")

print("데이터 그래프에 표시")
plt.figure(figsize= (8,8))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], marker='o', color='black', label='unclustered data')
 
plt.legend()
plt.title('plot of data points', size = 15)
plt.xlabel('Annual Income', size = 12)
plt.ylabel('Spending Score', size = 12)
plt.show()

print("raw level 구현")

# k개의 공집합 생성
# k=3일때만 가능, k= n개일땐 불가능

arr = [[],[],[]]
avr = [] #평균 담을 배열

all_data = np.array(data)
#print(all_data)

np.random.seed(42)
random_indices = np.random.choice(all_data.shape[0], size=3, replace=False)
random_samples = all_data[random_indices]

#print(random_samples)

for i in range(3): # 평균 담는 배열에 랜덤 평균값 삽입
    avr.append(random_samples[i])
    
print("임의의 평균1:" , np.array(avr[0]))
print("임의의 평균2:" , np.array(avr[1]))
print("임의의 평균3:" , np.array(avr[2]))

iteration = 100

def innerCompare(new, origin):
    for i in range(len(new)):
        isContain = False
        
        for j in range(len(origin)):
            new[i].sort()
            origin[j].sort()
            
            if new[i] == origin[j]:
                isContain = True
                break
            
        if not isContain:
            return False
        
    return True            
    
                  
def compare(new_arr, arr):
    for i in range(len(new_arr)):
        if len(new_arr[i]) != len(arr):
            return False
        
        if not innerCompare(new_arr[i], arr[i]):
            return False
    return True
        
               

for _ in range(iteration):
    #모든 데이터와 임의의 평균점 거리 측정
    
    new_arr = [[] for i in range(3)]
    
    for a_d in all_data:
        distances = [np.linalg.norm(a_d - cluster_mean) for cluster_mean in avr] 
        #print(distances) # 순서대로 각 avr[0], avr[1], avr[2]와의 거리 
        closest_cluster = np.argmin(distances) #argmin이 가장 가까운 거리에 있는 인덱스를 찾아주는 역할
        new_arr[closest_cluster].append(a_d) # 그 인덱스에 값 넣어줌.
    
    # early-stopping 
    if compare(new_arr, arr):
        arr = new_arr
        break
    
    # 평균 업데이트
    for i in range(3):
        avr[i] =  np.mean(new_arr[i], axis = 0)
    arr = new_arr 
              

print("업데이트 후 평균1:" , np.array(avr[0]))
print("업데이트 후 평균2:" , np.array(avr[1]))
print("업데이트 후 평균3:" , np.array(avr[2]))

#그래프 표시
plt.figure(figsize= (8,8))

arr1_t = np.array(arr[0]).T
plt.scatter(arr1_t[0], arr1_t[1], marker='o', label = 'cluster 1')

arr2_t = np.array(arr[1]).T
plt.scatter(arr2_t[0], arr2_t[1], marker='o', label = 'cluster 2')

arr3_t = np.array(arr[2]).T
plt.scatter(arr3_t[0], arr3_t[1], marker='o', label = 'cluster 3')

# avr
avr_t = np.array(avr).T
plt.scatter(avr_t[0], avr_t[1], marker='x', label = 'Centroids')

plt.title('Clustering Result')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()