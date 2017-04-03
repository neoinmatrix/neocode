# coding=utf-8
# the gamble game
# to predict the 1-38 38 number 
# if right then get the gamble_money*36
# how to gamble to get the money with 990 dollars
import random
import math
def gamble(money,number):
    return 0.0
    tmp=random.randint(1,38)
    if number==tmp:
        return money*36.0
    return 0.0
def suitable(setm=1000.0,money=990.0):
    for i in range(163):
        get=setm-money
        need=get/34.0
        money-=need
        if money<0.0:
            # return False
            # print "you are lost"
            break
        win=gamble(need,1)
        money+=win
        print "%i need %f win %f and left %f"%(i,need,win ,money)
        if money>=setm:
            break
    return money
        # get=1000.0-money
        # need=get/34.0
        # if need<=0:
        #     break

        # if money>1000.0:
        #     get=money-1000.0
        #     need=get/34.0
        # else:
        #     get=1000.0-money
        #     need=get/34.0
            # if need<=0:
            #     break
def test(num=10,setm=1000.0,money=990.0):
    fail=0
    success=0
    for i in range(num):
        getlast=suitable(setm,money)
        # print getlast
        if getlast>=setm:
            success+=1
    print "setm: %f success rate:%f "%(setm,float(success)/float(num))

# for i in range(1,100):
#     test(100,990.0+float(i)*10.0)

suitable()
print 1000.0/35.0