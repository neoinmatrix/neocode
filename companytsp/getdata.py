import torndb
import json

with open('../data/conf/db.conf') as json_file:
    config = json.load(json_file)
db=torndb.Connection(**config)

# sql="select * from lbs_district where district_id in "
# sql+=" (321282,440117,330802,340403,371502);"

# res=db.query(sql)
# print res
sql="select start_location as start,end_location as end,length,add_time "
sql+=" from orders where add_time>UNIX_TIMESTAMP('2018-01-21') "
sql+=" and district_id in (440117) order by add_time asc"
# sql+=" limit 0,10"

res=db.query(sql)

data={"start_lng":[],"start_lat":[],
"end_lng":[],"end_lat":[],
"length":[],"add_time":[]}

for v in res:
   data["start_lng"].append(float(v["start"].split(',')[0]))
   data["start_lat"].append(float(v["start"].split(',')[1]))
   data["end_lng"].append(float(v["end"].split(',')[0]))
   data["end_lat"].append(float(v["end"].split(',')[1]))
   data["length"].append(v["length"])
   data["add_time"].append(v["add_time"])
import pandas as pd
df=pd.DataFrame(data)
# print df.head()
df.to_csv('../data/440117.csv')
print "ok"
