import numpy as np
import numpy.matlib as npml
import tensorflow as tf
Ising_config=np.genfromtxt("Ising_N10.txt")
print(Ising_config)
print(Ising_config.shape)
print(Ising_config[0,0])
y_s=np.ndarray((20000,2))
x_s=np.ndarray((20000,100))
fin=open("Ising_N10.txt",'r')
cnt=0
for i in fin:
    if i[0]=='H':
        #y_s.append([1,0])
        y_s[cnt,:]=[1,0]
    elif i[0]=='L':
        #y_s.append([0,1])
        y_s[cnt,:]=[0,1]
    else:
        print("error in resolve input data/n")
    temp=i[2:202]
    temp=temp.split()
    for j in temp:
        j=int(j)
    x_s[cnt,:]=temp
    cnt+=1
print(y_s.shape)
print(x_s.shape)
print(y_s)
print(x_s)
x=tf.placeholder(tf.float32,[None,100])
W=tf.Variable(tf.zeros([100,2]))
b=tf.Variable(tf.zeros([2]))
y=tf.matmul(x,W)+b
y_=tf.placeholder(tf.float32,[None,2])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
# Train
tf.initialize_all_variables().run()
for i in range(5000):
    sess.run(train_step, feed_dict={x: x_s, y_: y_s})
    if i%50==0:
        print(sess.run(cross_entropy,feed_dict={x:x_s,y_:y_s}))
W_s=sess.run(W)
print(W_s)
