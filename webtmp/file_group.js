db.getCollection('api').group({
 keyf : function(doc){
  var date = new Date(doc.time);
  var dateKey = ""+date.getFullYear()+"-"+(date.getMonth()+1)+"-"+date.getDate();
  return {'day':dateKey}; //33
}, 
 initial : {"count":0}, 
 reduce : function Reduce(doc, out) {
    out.count +=1;
}
});











