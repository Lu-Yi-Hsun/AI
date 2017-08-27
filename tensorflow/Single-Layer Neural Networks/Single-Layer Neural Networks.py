import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(1,11,111).astype(np.float32)
#x_data=np.random.rand(100).astype(np.float32)
y_data=10*x_data+10.3


Weights=tf.Variable(tf.zeros([1]))

biases=tf.Variable(tf.zeros([1]))


y=Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))#用最小平方法 可以求出回歸直線 不能亂用
#************Optimizer*********
#****MomentumOptimizer
#optimizer=tf.train.MomentumOptimizer(0.1,0.2)#梯度下降法 Gradient descent


optimizer=tf.train.GradientDescentOptimizer(0.01)

train =optimizer.minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

#*********** for plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data,color="green")
plt.ion()
plt.show()

for step in range(2000):
    if step % 100 == 0:
        # 只是顯示參數
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(step, sess.run(Weights), sess.run(biases), sess.run(loss))
        #for plt

        lines = ax.plot(x_data, sess.run(y), 'r-', lw=1)
        plt.pause(0.01)


    sess.run(train)  # 真正訓練


