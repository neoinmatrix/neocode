# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
#y=2 * (x1) + (x2) + 3 
rate = 0.001
x_train = np.array([    [1, 2],    [2, 1],  
  [2, 3],    [3, 5],    [1, 3],    [4, 2],   
   [7, 3],    [4, 5],    [11, 3],    [8, 7]    ])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
x_test  = np.array([    [1, 4],    [2, 2],    [2, 5],    [5, 3],    [1, 5],    [4, 1]    ])

a = np.random.normal()
b = np.random.normal()
c = np.random.normal()

def h(x):
    return a*x[0]+b*x[1]+c

for i in range(100):
    sum_a=0
    sum_b=0
    sum_c=0
    for x, y in zip(x_train, y_train):
        for xi in x:
            sum_a = sum_a+ rate*(y-h(x))*xi
            sum_b = sum_b+ rate*(y-h(x))*xi
    a = a + sum_a
    b = b + sum_b
    c = c + sum_c
    plt.plot([h(xi) for xi in x_test])

print(a)
print(b)
print(c)

result=[h(xi) for xi in x_train]
print(result)

result=[h(xi) for xi in x_test]
print(result)

plt.show()