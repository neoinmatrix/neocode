# coding=utf-8
import time,platform
import os
import logging
import sys
import deal

svname='sv_logdeal'
logfile='./'+svname+'.log'
logpid='./'+svname+'.pid'

def testRunning():
    nname=deal.dealcmd()
    count=deal.deallog(nname)
    logging.info("insert num:%s"%str(count))

def createDaemon():  
    if os.path.exists(logpid):
        message = 'the '+svname+' process is running now'
        print message
        logging.info(message)
        os._exit(0) 
    try:
        if os.fork()> 0: 
            os._exit(0)
    except OSError, error:
        print 'fork failed: %d (%s)' % (error.errno, error.strerror)
        os._exit(1)    
    os.setsid()
    os.umask(0)
    with open(logpid, 'w') as f:
        pid=os.getpid()
        f.write(str(pid))
    # the main thread
    while True:
        testRunning()
        time.sleep(60*60*2)

if __name__ == '__main__': 
    # the main process command 
    for value in sys.argv[1:]:
        if value=="stop":
            with open(logpid, 'r') as f:
                pid=f.read()
            os.remove(logpid)
            output=os.popen('kill -9 '+pid)
            result=output.read()
            print result
            print "the process has stopped"
            os._exit(0)
        if value=="start":
            break
        if value=="remove":
            os.remove(logpid)
            os._exit(0)
        if value=="status":
            output=os.popen('ps -aux |grep '+svname)
            result=output.read()
            print result
            os._exit(0)

    if platform.system() == "Linux":
        logging.basicConfig(level=logging.DEBUG, \
            format='%(asctime)s %(levelname)-8s %(message)s', \
            datefmt='%Y-%m-%d %H:%M:%S', filemode='a+', filename=logfile)
        createDaemon()
    else:
        print "the system not support!"
        os._exit(0)
