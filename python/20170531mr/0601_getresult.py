# coding=utf-8
# labels=[0]*100000
# def datadeal():
#     with open('./dsjtzs_txfz_test1.txt','r') as f:
#         line='1'
#         while line:
#             line=f.readline()
#             if line=='':
#                 break
#             linecols=line.split(' ')
#             idx=int(linecols[0])-1
#             labels[idx]=1
#             # mouse=linecols[1]
#             # goal=linecols[2]
#             # label=int(linecols[3])

#             # labels[idx]=label
#             # goals[idx]=goal
#             # mouses[idx]=mouse.split(';')
#             # # marr=mouse.split(';')
#             # speeds[idx]=calc(mouse)
#     # print speeds
#     result=0
#     for v in labels:
#         result+=v
#     print result

# def main():
#     datadeal()
# with open('./result.txt','w') as f:
#     s=''
#     for i in range(1,100001):
#          s+=str(i)+"\n"
#     f.write(s)

# if __name__=='__main__':
#     main()
f=29.41
a=3.0*f/(500.0-2.0*f)
print a*100000