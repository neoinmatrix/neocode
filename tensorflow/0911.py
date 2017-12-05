# coding=utf-8
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import tensorflow as tf
import numpy as np

batch_size=8

w1=tf.Variable(tf.random_normal([2,3],stddev=1.,seed=1),dtype=tf.float32)
w2=tf.Variable(tf.random_normal([3,1],stddev=1.,seed=1),dtype=tf.float32)

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

# a=tf.nn.relu(tf.matmul(x,w1)+0.05)
# y=tf.nn.relu(tf.matmul(a,w2)+0.05)

cross_entropy= -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm=np.random.RandomState(1)
datasize=128
X=rdm.rand(datasize,2)
Y=[[float(x1+x2)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    # print sess.run(w1)
    # print sess.run(w2)

    fdict={x:X,y_:Y}
    total=sess.run(cross_entropy,feed_dict=fdict)

    steps=1001
    for i in range(steps):
        start=(i*batch_size)%datasize
        end=min(start+batch_size,datasize)
        # print start,end
        fdict={x:X[start:end],y_:Y[start:end]}
        sess.run(train,feed_dict=fdict)
        if i%200==0:
            fdict={x:X,y_:Y}
            total=sess.run(cross_entropy,feed_dict=fdict)
            print "steps %s cross_entropy is %s"%(i,total)

    # print sess.run(w1)
    # print sess.run(w2)

