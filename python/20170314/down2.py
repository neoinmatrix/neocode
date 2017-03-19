# coding=utf-8
import requests
import threading

class downloader:
    def __init__(self,url='',num=8):
        # the source of data url
        self.url='http://51reboot.com/src/blogimg/pc.jpg'
        if url!='':
            self.url=url
        # the threads of open
        self.num=num
        # the file name 
        self.name=self.url.split('/')[-1]
        # get heald
        r = requests.head(self.url)
        # print r.headers
        # headers length of data
        self.total = int(r.headers['Content-Length'])
        print 'total is %s' % (self.total)

    def get_range(self):
        # get the range of every segment
        ranges=[]
        offset = int(self.total/self.num)
        for i in  range(self.num):
            if i==self.num-1: # last one
                ranges.append((i*offset,''))
            else:
                ranges.append((i*offset,(i+1)*offset))
        # like [(0,12),(12,24),(25,36),(36,'')]
        return ranges

    def download(self,start,end):
        headers={'Range':'Bytes=%s-%s' % (start,end),'Accept-Encoding':'*'}
        res = requests.get(self.url,headers=headers)
        self.fd.seek(start)
        self.fd.write(res.content)

    def run(self):
        self.fd =  open(self.name,'w')
        thread_list = []
        n = 0
        for ran in self.get_range():
            start,end = ran
            print 'thread %d start:%s,end:%s'%(n,start,end)
            n+=1
            thread = threading.Thread(target=self.download,args=(start,end))
            thread.start()
            thread_list.append(thread)
        for i in thread_list:
            i.join()
        print 'download %s load success'%(self.name)

if __name__=='__main__':
    # url='http://pic6.huitu.com/res/20130116/84481_20130116142820494200_1.jpg'
    url='http://www.ipaotui.com/Public/Home/new/img/homepage/logo.png'
    down = downloader(url,1)
    down.run()
    # print down.get_range()