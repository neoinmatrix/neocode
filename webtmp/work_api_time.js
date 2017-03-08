var d1=new Date("2017-02-03T00:00:00")
var d2=new Date("2017-02-04T00:00:00")
var dt1=d1.getTime()/1000;
var dt2=d2.getTime()/1000;
db.api.aggregate(
    {$match:{
        time:{$gt:dt1,$lt:dt2},
//         elapse:{$lt:1},
        }},
    {$group:{
        _id:{
            "a":"$api"
        },
        //    elapse:{$exists:true},
        first:{$first:"$api"},
        max:{$max:"$elapse"},
        min:{$min:"$elapse"},
        count:{$sum:1},
        avg:{$avg:"$elapse"},
        sum:{$sum:"$elapse"},
      }},
    {$sort:{sum:-1}}
)