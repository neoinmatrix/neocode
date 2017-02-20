# coding=utf-8
import math
num = 1
while num<10000:
    print math.sqrt(num + 100)
    if math.sqrt(num + 100)-int(math.sqrt(num + 100)) == 0 and math.sqrt(num + 168)-int(math.sqrt(num + 168)) == 0:
        print(num)
        break
    num += 1