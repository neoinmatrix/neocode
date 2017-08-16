# coding=utf-8
import requests
from bs4 import BeautifulSoup
import numpy as np

def getdata(page=1):
    url = 'http://bdc.saikr.com/c/rl/34541?page='+str(page)
    doc = requests.get(url).text
    soup = BeautifulSoup(doc, "html.parser")
    table = soup.find('table')
    rows=table.find_all('tr')
    ranklist=[]
    for row in rows:
        cols=row.find_all('td')
        if len(cols)!=5:
            continue
        item=[]
        item.append(cols[0].contents)
        tmp=str(cols[1].find('div').contents)
        item.append(tmp.decode("unicode-escape"))
        tmp=str(cols[2].find('div').contents)
        item.append(tmp.decode("unicode-escape"))        
        item.append(cols[3].contents)
        item.append(cols[4].contents)
        ranklist.append(item)
    return ranklist


# data=getdata()
# print data.shape
if __name__=='__main__':
    data=getdata(1)
    # print data
    for i in range(2,6):
        tmp=getdata(i)
        data = np.vstack((data, tmp))
    # print data
    # np.save("./data.txt",data)

    # data=np.load('./data.txt.npy')
    college={}
    for v in data:
        name=v[2]
        # print name[3]
        # print name.decode('utf-8')
        college.setdefault(name,0)
        college[name]+=1
    # print college
    college=sorted(college.iteritems(), key=lambda d:d[1]) 
    # print college
    for k,v in college:
        print k,v
    # print college
 