# coding=utf-8 

def compare(file_a,file_b,samefile=''):
    with open(file_a,'r') as f:
        idxstr=f.read()
    rightidx=idxstr.split('\n')
    with open(file_b,'r') as f:
        idxstr=f.read()
    myidx=idxstr.split('\n')
    arr=[0]*100001
    for v in rightidx:
        if v=='':
            continue
        idx=int(v)
        arr[idx]+=1
    for v in myidx:
        if v=='':
            continue
        idx=int(v)
        arr[idx]+=1
    count=0
    str_result=''
    for i in range(1,100001):
        if arr[i]==2:
            # print i
            count+=1
            str_result+=str(i)+"\n"
    print "file_a: ",len(rightidx)
    print "file_b: ",len(myidx)
    print "samenum:",count
    if samefile!='':
        with open(samefile,'w') as f:
            f.write(str_result)
    print "over==============="

# file_b='./data/94.txt'
file_a='./data/BDC1236_20170711_p.txt' # 95.70 19901 19083 
# file_a='./data/BDC1236_20170709.txt'   # 95.69 19256 18710 
# file_a='./data/9448.txt'               # 94.80 20045 18985 

# file_b='0712tmp.txt'
# file_b='0713tmp.txt'
# file_b='./data/BDC1236_20170709.txt'   # 95.69 19256 18710 
file_b='./data/16/m_0.txt' 
compare(file_a,file_b) 
file_b='./data/17/m_0.txt'  
compare(file_a,file_b)

compath='./data/17/'
for i in range(1,6): 
    file_b=compath+'r_%d.txt'%i
    file_s=compath+'s_%d.txt'%i
    compare(file_a,file_b,file_s)




import datadeal
# print "====="
# print datadeal.calcScoreRerve(0.9973,20000) #19946
# print datadeal.calcScoreRerve(0.9570,19901) #19083
# print datadeal.calcScoreRerve(0.9569,19256) #18710
# print datadeal.calcScoreRerve(0.9480,20045) #18985
# print datadeal.calcScoreRerve(0.9270,18683) #17807
# print datadeal.calcScoreRerve(0.9489,19422) #18648
