import os

def eachFile(filepath):
    dirs =  os.listdir(filepath)
    dirs_arr=[]
    for v in dirs:
        child = os.path.join('%s%s' % (filepath, v))
        if os.path.isdir(child):
            dirs_arr.extend(eachFile(child+"/"))
        else:
            dirs_arr.append(child)
    return dirs_arr

if __name__ == '__main__':
    res=eachFile('./')
    # print res
    for v in res:
        # print v.enco
        print v.decode('utf-8')
