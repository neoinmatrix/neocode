### 均值滤波
 
#### 题目：
- 均值滤波器
- 对于一维数组进行均值滤波，
- 就是每个点的数据为，以此点为中心半径为r的数组单元中
- 取值并且，求平均值，以过滤数据中波动大的数据

#### 要求:
- C 语言实现代码
- Cuda 语言实现代码
- shared memory的使用
- 误差小数据访问不会越界
- 计算运行时间和计算误差
- 支持大数据的处理

#### thinking: 

- the data of margin side can be dealed by this  (i-j+n)%n 
shaped the array  like circle 

- in the same block ,the threads  visit the  data range in [r-i r r+i] so copy global memory to shared memory to boost the speed

- the shared memory is 48KB  so the num of radius    3*r<= (48KB/4B)  =>  r <= 4K 


#### 测试数据:  ()滤波数组大小 滤波半径 测试打印（0不打印 1打印）
10 3 1
100000 100 0
100000 50 0
