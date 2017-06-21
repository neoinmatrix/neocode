# coding=utf-8

# generate all number of data and recorded
def saveallresult():
    datadeal()
    with open('./result.txt','w') as f:
        s=''
        for i in range(1,100001):
             s+=str(i)+"\n"
        f.write(s)

if __name__=='__main__':
    main()

# I choose the all 100,000 data get the scores 29.41
# so with s= (5*P*R)/(2*P+3*R) P=1 
# get the R and number of machinedata
def getMachineNumber():
    f=29.41
    a=3.0*f/(500.0-2.0*f)
    print a
    print a*100000
getMachineNumber()