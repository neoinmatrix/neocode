# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

'''蒙特卡罗方法求函数 y=x^2 在[0,1]内的定积分（值）'''
def f(x):
    return x**2

# 投点次数
n = 10000

# 矩形区域边界
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0     

# 在矩形区域内随机投点
x = np.random.uniform(x_min, x_max, n) # 均匀分布
y = np.random.uniform(y_min, y_max, n)

# 统计 落在函数 y=x^2图像下方的点的数目
res = sum(np.where(y < f(x), 1, 0))

# 计算 定积分的近似值（Monte Carlo方法的精髓：用统计值去近似真实值）
integral = float(res) / float(n)

print('integral: ', integral)

# 画个图看看
fig = plt.figure() 
axes = fig.add_subplot(111) 
axes.plot(x, y,'ro',markersize = 1)
plt.axis('equal') # 防止图像变形

axes.plot(np.linspace(x_min, x_max, 10), f(np.linspace(x_min, x_max, 10)), 'b-') # 函数图像
#plt.xlim(x_min, x_max)

plt.show()