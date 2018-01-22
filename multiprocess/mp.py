# coding=utf-8
from multiprocessing import Process

def f(name):
    print 'hello', name

if __name__ == '__main__':
    p = Process(target=f, args=('bob',)) # 新建一个子进程p，目标函数是f，args是函数f的参数列表
    p.start() # 开始执行进程
    print "here"
    p.join() # 等待子进程结束