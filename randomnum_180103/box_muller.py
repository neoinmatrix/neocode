import numpy as np  

def boxmuller():
    summa = 1
    size = 1
    x = np.random.uniform(size=size)  
    y = np.random.uniform(size=size)  
    z = np.sqrt(-2 * np.log(x)) * np.cos(2 * np.pi * y)  
    q =  z * summa
    return q
  
# print(boxmuller()[0])

def main():
    a=[]
    for i in range(10000):
        a.append(boxmuller())
    a=np.array(a)
    # print a
    import matplotlib.pyplot as plt
    plt.hist(a,bins=86)
    # plt.hist
    # plt.plot(range(2000),a)
    plt.show()

main()