import mysql.connector
class Model(object):
	__instance   	= None
	__host	   		= None
	__port	   		= None
	__user	  		= None
	__password 		= None
	__database   	= None
	__charset	 	= None
	__session		= None
	__connection 	= None
	__sql			= None
	def __init__(self, host='localhost',port='3306', user='root', password='', database='',charset='utf8'):
		self.__host	 = host
		self.__port	 = port
		self.__user	 = user
		self.__password = password
		self.__database = database
		self.__charset  = charset

	def __open(self):
		try:
			cnx = mysql.connector.connect(
				host=self.__host,port=__port, user=self.__user, password=self.__password,
				database=self.__database,charset=self.__charset)
			self.__connection = cnx
			self.__session	= cnx.cursor()
		except mysql.Error as e:
			print "Error %d: %s" % (e.args[0],e.args[1])

	def __close(self):
		self.__session.close()
		self.__connection.close()

	def where(self,table):
		if type(table)==dict:
			tmp=[]
			for k,v in table.iteritems():
				tmp.append(" `%s` = '%s' "%(k,v))
			self.__where=" and ".join(tmp)

	def table(self,table):
		self.__table=table

	def select(self, table, where=None, *args, **kwargs):
		result = None
		query = 'SELECT '
		keys = args
		values = tuple(kwargs.values())
		l = len(keys) - 1

		for i, key in enumerate(keys):
			query += "`"+key+"`"
			if i < l:
				query += ","
		## End for keys

		query += 'FROM %s' % table

		if where:
			query += " WHERE %s" % where
		## End if where

		self.__open()
		self.__session.execute(query, values)
		number_rows = self.__session.rowcount
		number_columns = len(self.__session.description)

		if number_rows >= 1 and number_columns > 1:
			result = [item for item in self.__session.fetchall()]
		else:
			result = [item[0] for item in self.__session.fetchall()]
		self.__close()

		return result
	## End def select

config={
	'host':'ip',
    'user':'user',
    'password':'password',
    'port':3306 ,
    'database':'test',
    'charset':'utf8'
}
m=Model(**config)
where={
	"user_id":11,
	"name":'neo',
}
m.table('tbx').where(where)
for property, value in vars(m).iteritems():
    print property, ": ", value