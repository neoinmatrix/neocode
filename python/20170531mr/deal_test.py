# coding=utf-8
# import deal 
# print deal.trainset()
import numpy as np
import datadeal as dd
import deal as dl

def getMouseData(mouse):
    tmp=mouse.split(' ')
    idx=tmp[0]
    # print tmp
    mouse=dd.posdata(tmp[1])
    # print mouse
    mouse=dl.getSingle(mouse)
    return idx,mouse
# line="1458 332,2503,214;339,2503,274; 1088.0,189"
# idx,mouse=getMouseData(line)

def testdata():
    count=0
    clf=dl.trainset()
    with open('./dsjtzs_txfz_test1.txt','r') as f:
        line='1'
        barr=[]
        barrs=''
        while line:
            line=f.readline()
            if line=='':
                break
            count+=1
            try:
                idx,mouse=getMouseData(line)
                if idx=="27":
                    print line
                    break
            except Exception as e:
                print line 
            
            r=clf.predict(np.array([mouse]))
            if r[0]<1:
                # print idx
                barr.append(idx)
                barrs+="%s\n"%idx
            if count%1000==0:
                print count
        print len(barr)
        print count
    with open('./testfind.txt','w') as fx:
        fx.write(barrs)

# testdata()
mouse="27 346,2490,52;367,2529,1810;381,2529,1843;409,2529,1879;437,2529,1924;465,2529,1960;493,2529,1999;507,2529,2020;535,2529,2047;577,2529,2092;619,2529,2122;661,2529,2158;710,2529,2182;759,2529,2599;808,2529,2638;857,2529,3331;906,2529,4006;955,2529,4051;990,2529,4078;1025,2529,4108;1060,2529,4156;1095,2529,4186;1123,2529,4216;1158,2529,4282;1193,2529,4342;1228,2529,4423;1277,2529,4471;1298,2529,4549; 1063.5,189"
tmp=mouse.split(' ')
idx=tmp[0]
# print tmp
mouse=dd.posdata(tmp[1])
import draw 
draw.draw(mouse)
print mouse