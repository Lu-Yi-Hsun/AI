import tensorflow as tf
import numpy as np

# create data
ca_range=20
#**********************************************
with open("80211ax.txt") as f:
    file = f.read()
data=file.split("\n")
rss=[]
datarate=[]
angle=[]
node=[]
dalay=[]
for strr in data:
    strr_arry = strr.split(" ")
    if strr_arry[0].isdigit():

        #input rss
        print(strr_arry)
        print("\n")
        rss.append(strr_arry[8])
        #input data rate
        datarate.append(strr_arry[0])
        #input angle 角度換算 換成0~180
        strr_arry[6] = abs(int(strr_arry[6]))
        while strr_arry[6] > 180:
            strr_arry[6] = 360 - strr_arry[6]
        angle.append(strr_arry[6])
        node.append(strr_arry[7])
        dalay.append(strr_arry[2])

rss=np.array(rss).astype(np.float32)
dalay=np.array(dalay).astype(np.float32)
datarate=np.array(datarate).astype(np.float32)
node=np.array(node).astype(np.float32)

#dalay=np.random.rand(1000).astype(np.float32)
#************************************************
a1 = tf.Variable(tf.random_uniform([1], -1*ca_range, ca_range))

a2 = tf.Variable(tf.random_uniform([1], -1*ca_range, ca_range))
a3 = tf.Variable(tf.random_uniform([1], -1*ca_range, ca_range))

a4 = tf.Variable(tf.random_uniform([1], -1*ca_range, ca_range))


biases = tf.Variable(tf.zeros([1]))

y = a1*rss+a2*datarate+a3*angle+a4*node+biases

loss = tf.reduce_mean(tf.square(y-dalay))

optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)


# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好



sess = tf.Session()
sess.run(init)          # Very important
print("rss datarate angle node biases")
l2=0
l1=11
step=0


while (l1!=l2):
    _,los=sess.run([train,loss])
    if step%2==0:
        l1=los
    elif step%2!=0:
        l2=los
    if step % 20 == 0:
        print(step, sess.run(a1),sess.run(a2),sess.run(a3),sess.run(a4),sess.run(biases))


        print(los)

    step=step+1