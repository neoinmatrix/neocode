# monitor shadowsocks connections with python service
I have a shadowsocks server in bandwagonhost
and I want to monitor who connect and connect what 
and I use python to analyst the log from shadowsock
in default position /var/log/shadowsocks.log
I use mysql db base to restore the data 
# the dependences 
. pip install mysql-python
. pip install torndb
# the command to service
. python sv_logdeal.py start
. python sv_logdeal.py status 
. python sv_logdeal.py stop

### designed by neo in 2017-05-30