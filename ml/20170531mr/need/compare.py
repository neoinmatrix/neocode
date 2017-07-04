# coding 
with open('../data/94.txt','r') as f:
    idxstr=f.read()
rightidx=idxstr.split('\n')
print len(rightidx)

with open('../data/0704tmp.txt','r') as f:
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
for i in range(1,100001):
    if arr[i]==2:
        count+=1
print count