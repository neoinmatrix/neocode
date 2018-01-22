# coding=utf-8
import multiprocessing
import time

def func(msg):
    print multiprocessing.current_process().name + '-' + msg
    for i in range(100000):
        for j in range(100000):
            i+j
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=3) # 创建4个进程
    for i in xrange(10):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))
        # pool.apply(func, (msg, ))
    pool.close() # 关闭进程池，表示不能在往进程池中添加进程
    pool.join() # 等待进程池中的所有进程执行完毕，必须在close()之后调用
    print "Sub-process(es) done."