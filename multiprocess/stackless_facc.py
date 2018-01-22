import stackless
 
dic = {}
 
def factorial(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return task(n-1) + task(n-2)
 
 
def task(n):
    if str(n) in dic:
        return dic[str(n)]
    chann = stackless.channel()
    stackless.tasklet(compute)(n,chann)
    result = chann.receive()
    dic[str(n)] = result
    return result
 
def compute(n,chann):
    return chann.send(factorial(n))
 
 
print factorial(100000)