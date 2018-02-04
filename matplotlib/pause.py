import numpy as np  
import matplotlib.pyplot as plt  
  
plt.axis([0, 10, 0, 1])  
plt.ion()  
  
for i in range(10):  
    y = np.random.random()  
    plt.scatter(i, y)  
    plt.pause(0.5)  
  
plt.ioff()
plt.show()
# while True:  
#     plt.pause(0.05)  