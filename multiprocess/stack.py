import sys
sys.setrecursionlimit(100000)
 
dic = {}
 
def factorial(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    elif str(n) in dic:
        return dic[str(n)]
    else:
        a = factorial(n-1)
        if str(n-1) not in dic:
            dic[str(n-1)] = a
        b = factorial(n-2)
        if str(n-2) not in dic:
            dic[str(n-2)] = a
        return a + b
 
 
print factorial(99999)
