'''
Basic Operations example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input


a = tf.placeholder(tf.float32)


# Define some operations
add = tf.log(a)


# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %f" % sess.run(add, feed_dict={a: 0}))
