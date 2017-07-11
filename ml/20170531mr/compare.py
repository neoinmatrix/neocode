# coding 
path='./data/'
with open(path+'BDC1236_20170711_p.txt','r') as f: # 95.70 19083
    idxstr=f.read()
rightidx=idxstr.split('\n')
print len(rightidx)

with open(path+'0711tmp.txt','r') as f:
    idxstr=f.read()
myidx=idxstr.split('\n')
print len(myidx)

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
        count+=1
        str_result+=str(i)+"\n"
print count
# with open('../data/0709_2f_filter.txt','w') as f:
#     f.write(str_result)

# import datadeal
# print "====="
# print datadeal.calcScoreRerve(0.9570,19901)
# # print datadeal.calcScoreRerve(0.9570,19902)
# # print datadeal.calcScoreRerve(0.9973,20000)