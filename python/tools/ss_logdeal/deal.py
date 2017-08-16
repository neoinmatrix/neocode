# coding=utf-8
import time
import datetime
import dbop
import os
import shutil
import logging

def deallog(logfile='/var/log/shadowsocks.log'):
    s=''
    with open(logfile) as f:
        s=f.read()
    sarr=s.split("\n")
    count=0
    for v in sarr:
        if v.find("INFO")<0:
            continue
        if v.find("from")<0:
            continue
        varr=v.split(" ")
        if varr[8].find(":")>0:
            dst=varr[8].split(":")[0]
            dst_port=varr[8].split(":")[1]
        else:
            dst=varr[8]
            dst_port=""
        if varr[10].find(":")>0:
            req=varr[10].split(":")[0]
            req_port=varr[10].split(":")[1]
        else:
            req=varr[10]
            req_port=""
        timevalue=time.mktime(time.strptime(varr[0]+" "+varr[1],'%Y-%m-%d %H:%M:%S'))
        value={"time":timevalue,\
        "type":varr[2],"dst":dst,"dst_port":dst_port,\
        "req":req,"req_port":req_port,"add_time":time.mktime(datetime.datetime.now().timetuple())}
        # print value
        count+=1
        dbop.insert(value)
    return count

def dealcmd():
    # change name
    logpath='/var/log/'
    oname="shadowsocks.log"
    nname=str(time.mktime(datetime.datetime.now().timetuple()))[:-2]+".log"
    os.rename(logpath+oname,logpath+nname)
    # restart service
    restart_cmd='''
    /usr/bin/ssserver -s 0.0.0.0 -p `cat /root/.kiwivm-shadowsocks-port` -k `cat /root/.kiwivm-shadowsocks-password` -m `cat /root/.kiwivm-shadowsocks-encryption` --user nobody --workers 2 -d restart
    '''
    output=os.popen(restart_cmd)
    result=output.read()
    logging.info("result:%s"%result)
    # print result
    # mv here 
    shutil.move(logpath+nname,"./"+nname)
    logging.info("result:%s"%result)
    return "./"+nname

if __name__=="__main__":
    pass
    # dealcmd()
    # deallog("./1496191680.log")
