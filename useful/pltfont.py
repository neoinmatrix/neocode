# coding=utf-8
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rcParams['font.sans-serif'] = ['simhei']
# mpl.rcParams['axes.unicode_minus'] = False
plt.plot(range(10),range(10))
plt.rcParams['font.sans-serif'] = ['simhei'] # 指定默认字体
plt.rcParams['font.serif'] = ['simhei'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.xlabel(u"横轴")
plt.show()