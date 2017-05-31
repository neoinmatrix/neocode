import torndb
import json

# write db info in db.conf  never in the pyfile
with open('db.conf') as json_file:
    config = json.load(json_file)
db=torndb.Connection(**config)

def insert(data):
    sql='insert into ss_info'
    field=''
    value=''
    for k in data:
        field+=k+","
        value+="'%s',"%(str(data[k]))
    field=field[:-1]
    value=value[:-1]
    sql="insert into ss_info (%s)value(%s)"%(field,value)
    gid=db.execute(sql)
    if gid>0:
        return gid
    return -1

if __name__=="__main__":
    # data=db.query("select * from ss_info limit 0,10")
    value={"time":1231231,\
    "type":"info","dst":"133.222.222.112","dst_port":1331,\
    "req":"32.222.222.115","req_port":2442,"add_time":123123}
    # print value
    print insert(value)
    