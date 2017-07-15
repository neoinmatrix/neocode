# coding=utf-8 
def compare(file_a,file_b,samefile='',path='./data/'):
    with open(path+file_a,'r') as f:

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

# 95.70 19083 
file_a='BDC1236_20170711_p.txt'
# file_b='0712tmp.txt'
# file_b='0713tmp.txt'
compath='./data/15/'
for i in range(1,6): 
    file_b=compath+'r_%d.txt'%i
    file_s=compath+'s_%d.txt'%i
    compare(file_a,file_b,file_s)
# file_b='0714_pos.txt'
# compare(file_a,file_b,'0714_pos_s.txt')
# file_b='0714_class.txt'
# compare(file_a,file_b,'0714_class_s.txt')
# file_b='0714_class0.txt'
# compare(file_a,file_b,'0714_class0_s.txt')

# import datadeal
# print "====="
# print datadeal.calcScoreRerve(0.9570,19901)
# # print datadeal.calcScoreRerve(0.9570,19902)
# # print datadeal.calcScoreRerve(0.9973,20000)