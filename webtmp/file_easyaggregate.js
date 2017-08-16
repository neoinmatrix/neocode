var d1=new Date("2016-12-23")
var d2=new Date("2016-12-24")
var dt1=d1.getTime()/1000;
var dt2=d2.getTime()/1000;

db.api.aggregate(
    {$match:{time:{$gt:dt1,$lt:dt2}}},
    {$group:{_id:"$api",count:{$sum:1}}}
)
    