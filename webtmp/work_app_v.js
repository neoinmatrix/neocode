var d1=new Date("2017-07-24T00:00:00")
var d2=new Date("2017-07-25T00:00:00")
var dt1=d1.getTime()/1000;
var dt2=d2.getTime()/1000;
db.api.aggregate(
    {$match:{
        time:{$gt:dt1,$lt:dt2},
   
        }},
    {$group:{
        _id:{
            "v":"$app_v",
            "a":"$api"
        },
        v:{$first:"$app_v"},
        a:{$first:"$api"},
        count:{$sum:1}
      }},
    {$sort:{count:-1}}
)