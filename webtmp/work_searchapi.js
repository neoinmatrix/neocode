var d1=new Date("2017-07-26T00:00:00")
var d2=new Date("2017-07-27T00:00:00")
var dt1=d1.getTime()/1000;
var dt2=d2.getTime()/1000;
var tmp=db.getCollection('api').find(
{
    api:"/index.php?m=Api&c=Order&a=getOrderList",
    time:{$gt:dt1,$lt:dt2},
//     elapse:{$gt:1},
}
).limit(10).sort({time:-1})
var format=true
if(format == true){
    tmp.forEach(function(row, index, array){
        for(i in row){
            print(i+":"+row[i])
        }
        print('\n =========================== \n')
    })
}else{
     tmp.forEach(function(row, index, array){
        print(row)
    })
}