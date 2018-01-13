import numpy as np
import time
import multiprocessing

def execute(idx,num):
    for i in range(num):
        print "%d,%d"%(idx,i)
        time.sleep(1)
    print "over"

if __name__=="__main__":
    pool = multiprocessing.Pool(processes=3)
    for idx in range(4): 
        pool.apply_async(execute, (idx,10))
    pool.close()
    pool.join()
    print "over"