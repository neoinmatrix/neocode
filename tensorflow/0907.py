import tensorflow as tf 
import numpy as np 

coff=np.array([[1.],[-20.],[100.]])
w=tf.Variable(0.0,dtype=tf.float32)
x=tf.placeholder(tf.float32,[3,1])
cost=x[0][0]*w**2+x[1][0]*w+x[2][0]

train=tf.train.GradientDescentOptimizer(0.1).minimize(cost)
init=tf.global_variables_initializer()
session=tf.Session()
session.run(init)
session.run(train,feed_dict={x:coff})
for i in range(100):
    session.run(train,feed_dict={x:coff})
    
print session.run(w)
