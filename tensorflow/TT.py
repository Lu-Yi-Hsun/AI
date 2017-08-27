import tensorflow as tf
import numpy as np
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return (outputs,Weights,biases)
x_data_size=100
x_data=np.linspace(1,11,x_data_size).astype(np.float32)[:,np.newaxis]

print(x_data)
y_data=12.3*x_data+33
y,Weights,biases=add_layer(x_data,1,x_data_size)

loss=tf.reduce_mean(tf.square(y-y_data))


optimizer =tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train)

        print("loss:"+str(sess.run(loss)))
        print(sess.run(Weights)[0][0])