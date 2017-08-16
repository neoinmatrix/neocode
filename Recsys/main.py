# coding=utf-8
import numpy as np
import time
import dataset

if __name__=='__main__':
    # print 'hello world'
    np.set_printoptions(linewidth=200,formatter={'float':lambda x: "%8.3f"%float(x)})
    start=time.clock()
    wsdata=dataset.Wsdata()
    wsdata.getRtMatrix()
    wsdata.getTpMatrix()
    end=time.clock()
    # print "data is ready! use: %s s"%(end-start)
    print wsdata.rt_matrix[0,0:10]
    print wsdata.tp_matrix[0,0:10] 
    print "====================="
    print wsdata.rt_matrix[1,0:10]
    print wsdata.tp_matrix[1,0:10]
    # wsdata.getUserList()
    # wsdata.getWsList()
