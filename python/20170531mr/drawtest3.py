import matplotlib.pyplot as plt
import datadeal as dd
import deal as dl

def getMouseData(mouse):
    tmp=mouse.split(' ')
    idx=tmp[0]
    # print tmp
    mouse=dd.posdata(tmp[1])
    # print mouse
    # mouse=dl.getSingle(mouse)
    return idx,mouse

def testdata():
    count=0
    with open('./data/dsjtzs_txfz_test1.txt','r') as f:
        line='1'
        xarr=[]
        yarr=[]
        while line:
            line=f.readline()
            if line=='':
                break
            count+=1
            try:
                idx,mouse=getMouseData(line)
            except Exception as e:
                print line 
            # print mouse[0]
            # exit()
            if mouse[1][0]<1200 or mouse[0][0]<0:
                continue
            xarr.append(mouse[0][0])
            yarr.append(mouse[1][0])
            if count%1000==0:
                print count
            if count>10000:
                break
        plt.scatter(xarr,yarr)
        plt.show()
testdata()