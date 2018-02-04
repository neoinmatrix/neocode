# coding=utf-8
from common.DemonClass import *
import json
import os
class ClientDaemon(CDaemon):
    def __init__(self, name, conf_file, save_path, stdin=os.devnull, stdout=os.devnull, stderr=os.devnull, home_dir='.', umask=022, verbose=1):
        CDaemon.__init__(self, save_path, stdin, stdout, stderr, home_dir, umask, verbose)
        self.name = name #派生守护进程类的名称
        self.conf_file=conf_file

    def run(self, output_fn, **kwargs):
        fd = open(output_fn, 'w')
        if os.path.exists(conf_file)==False:
            print "Error conf file not exists"
            return 

        with open(self.conf_file) as json_file:
            self.config = json.load(json_file)

        while True:
            # line = time.ctime() + '\n'
            # fd.write(line)
            # fd.flush()
            time.sleep(1)
        fd.close()


if __name__ == '__main__':
    help_msg = 'Usage: python %s <start|stop|restart|status>' % sys.argv[0]
    if len(sys.argv) != 2:
        print help_msg
        sys.exit(1)
    p_name = 'TspRS'
    pid_fn = '/tmp/%s.pid'%p_name
    # log_fn = '/tmp/%s.log'%p_name 
    err_fn = '/tmp/%s.err.log'%p_name
    log_fn = '/data/neocode/companytsp/tsp.log' #守护进程日志文件的绝对路径
    # err_fn = '/tmp/daemon_class.err.log' #守护进程启动过程中的错误日志,内部出错能从这里看到
    conf_file="/data/neocode/data/conf/tspx.conf"
    cD = ClientDaemon(p_name, conf_file, pid_fn, stdout=log_fn,stderr=err_fn, verbose=1)

    if sys.argv[1] == 'start':
        cD.start(log_fn)
    elif sys.argv[1] == 'stop':
        cD.stop()
    elif sys.argv[1] == 'restart':
        cD.restart(log_fn)
    elif sys.argv[1] == 'status':
        alive = cD.is_running()
        if alive:
            print 'process [%s] is running ......' % cD.get_pid()
        else:
            print 'daemon process [%s] stopped' %cD.name
    else:
        print 'invalid argument!'
        print help_msg