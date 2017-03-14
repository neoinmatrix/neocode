# coding=utf-8
# the python program to ananlyst the paper 
# and download the file (like *.csv *.data)
# to save in the local path 
# coded by neo  2017-03-14

import urllib
import urllib2
import re

def getdoc(url):
    req = urllib2.Request(url) 
    con = urllib2.urlopen(req)
    doc = con.read()
    con.close()
    return doc
    
def dealinfo(doc):
    rex = r'<a.*? href="(.*?)".*?>.*?</a>'
    tmp_data=re.findall(rex,doc)
    data=[]
    for row in tmp_data:
        if row.find(".dat")>0 or row.find(".csv")>0:
            data.append(row)
    return data

def download(url,file,savepath='./'):
    resource=url+file
    savefile=savepath+file
    urllib.urlretrieve(resource, savefile)

if __name__=="__main__":
    url = 'http://www.cs.cmu.edu/~chongw/data/citeulike/' 
    doc = getdoc(url)
    data=dealinfo(doc)
    # download(url,data[0])
    for row in data:
        print row
        # download(url,row)
        print "finished ============"

