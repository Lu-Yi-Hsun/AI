from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math
import  random
PI=3.14159
def to_gray(img):
    w, h,_ = img.shape
    ret = np.empty((w, h), dtype=np.uint8)
    retf = np.empty((w, h), dtype=np.float)
    imgf = img.astype(float)
    retf[:, :] = ((imgf[:, :, 1] + imgf[:, :, 2] + imgf[:, :, 0])/3)
    ret = retf.astype(np.uint8)
    return ret
#轉換圖片核心
def radia_transform(im,m,n):
    shape = im.shape
    new_im = np.zeros(shape)
    width = shape[0]
    height = shape[1]
    lens=len(shape)
    for i in range(0,width):
        xita = 2*PI*(i)/width
        for a in range(0,height):
            x = (int)(math.floor(a * math.cos(xita)))
            y = (int)(math.floor(a * math.sin(xita)))
            new_y = (int)(m+x)
            new_x = (int)(n+y)
            if new_x>=0 and new_x<width and new_y>=0 and new_y<height:
                if lens == 3:
                    #做有彩度rgb
                    new_im[a, i, 0] = im[new_y, new_x, 0]
                    new_im[a, i, 1] = im[new_y, new_x, 1]
                    new_im[a, i, 2] = im[new_y, new_x, 2]
                else:
                    #只有黑白
                    new_im[a, i] = im[new_y, new_x]

    return new_im


#im=to_gray(im)



#im = io.imread('e.png')
# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images

# 印出來看看


for i in range(10):
    im = np.reshape(x_train[random.randint(0,55000), :], (28, 28))
    h = im.shape[0]
    w = im.shape[1]
    new_im1 = radia_transform(im, (int)(w / 2), (int)(h / 2))

    new_im2 = radia_transform(im, (int)(w *random.uniform(0,1)), (int)(h*random.uniform(0,1)))

    new_im3 = radia_transform(im, (int)(w * 0.5), (int)(h * 0.75))
    plt.figure(num='astronaut',figsize=(8,8))

    plt.subplot(2,2,1)
    plt.title('origin image')
    plt.imshow(im,plt.cm.gray)

    plt.subplot(2,2,2)
    plt.title('0.5')
    plt.imshow(new_im1,plt.cm.gray)
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.title('0.25')
    plt.imshow(new_im2,plt.cm.gray)
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.title('0.75')
    plt.imshow(new_im3,plt.cm.gray)
    plt.axis('off')

    plt.show()