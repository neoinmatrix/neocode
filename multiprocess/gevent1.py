import gevent  
   
def foo():  
    print('Running in foo')  
    gevent.sleep(0)  
    print('Explicit context switch to fooagain')  
   
def bar():  
    print('Explicit context to bar')  
    gevent.sleep(0)  
    print('Implicit context switch back tobar')  
   
gevent.joinall([  
    gevent.spawn(foo),  
    gevent.spawn(bar),  
]) 