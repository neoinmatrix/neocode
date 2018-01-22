#!/usr/bin/env python
#coding=utf8
 
import os,time
 
#创建子进程之前声明的变量
source = 10
 
try:
    pid = os.fork()
 
    if pid == 0: #子进程
        print "this is child process."
        #在子进程中source自减1
        source = source - 1
        time.sleep(10)
        print "subover"
    else: #父进程
        print "this is parent process."
        time.sleep(10)
        print "parent over"
 
    print source
except OSError, e:
    pass