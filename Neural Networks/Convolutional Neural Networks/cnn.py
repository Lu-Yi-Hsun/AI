from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):

    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs,Weights,biases

def nu(x_shape,y_shape):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, x_shape])
    ys = tf.placeholder(tf.float32, [None, y_shape])
    # add hidden layer 隱藏層
    neural_node=3000
    l1 ,Weights1,biases1= add_layer(xs, x_shape, neural_node, activation_function=tf.nn.relu)



    # add output layer 輸出層
    prediction ,Weights2,biases2= add_layer(l1, neural_node, y_shape, activation_function=tf.nn.softmax)
    kkk=tf.log(prediction)

    # the error between prediction and real data
    loss = tf.reduce_mean(-tf.reduce_sum(ys * prediction, reduction_indices=1))
    #loss=tf.square(ys-prediction)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # important step
    config = tf.ConfigProto(
      device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    with tf.name_scope('Accuracy'):
      correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
      acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('Accuracy', acc)
    return  sess,train_step,xs,ys,prediction,acc,loss,kkk



# plot the real data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_test = mnist.test.images
y_test = mnist.test.labels

# start
sess,train_step,xs,ys,prediction,acc,loss,kkk=nu(784,10)
for step in range(10000):

    # training

    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    #if step % 50 ==0:
        #print(sess.run(loss, feed_dict = {xs: batch_xs, ys: batch_ys}))

print("Accuracy: ", sess.run(acc, feed_dict={xs: x_test, ys: y_test}))



