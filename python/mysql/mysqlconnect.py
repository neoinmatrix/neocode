import mysql.connector
config={
	'host':'ip',
    'user':'user',
    'password':'passowrd',
    'port':3306 ,
    'database':'db_xx',
    'charset':'utf8'
}
try:
	cnn=mysql.connector.connect(**config)
except mysql.connector.Error as e:
	print('connect fails!{}'.format(e))
try:
	cursor=cnn.cursor()
	sql_query='select * from tba'
	cursor.execute(sql_query)
	for id,name in cursor:
		print ('%s %s'%(name,id))
except mysql.connector.Error as e:
	print('query error!{}'.format(e))
finally:
	cursor.close()
	cnn.close()