var d1 =new Date('2016-08-23');
var d2 =new Date('2016-08-24');
var dt1=d1.getTime()/1000;
var dt2=d2.getTime()/1000;

db.api.mapReduce(
    map = function() {
         emit(this.time, 1);
    }, 
    reduce = function(key, values) {
//         var cnt = 0;
//         values.forEach(function(value) {
//             cnt += value;
//         })
       
        return values.length;
    }, 
    query={
        out: "api_tmp", 
        query: {time: {$gt: dt1, $lt: dt2}},
        sort: {time: 1}
    }
)

        
        
        
        
        
        