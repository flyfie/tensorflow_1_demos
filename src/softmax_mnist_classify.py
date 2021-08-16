import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.sparse import csr_matrix


def dense_y(y):
    cnt = len(y)
    data = np.ones(cnt)
    # row = np.array(range(cnt))
    row = np.arange(cnt)

    return csr_matrix((data, (row, y)), shape=(cnt, 10)).toarray()


logs_path = 'log_mnist_softmax'
batch_size = 100
learning_rate = 0.5
training_epochs = 20

mnist = input_data.read_data_sets("data")

# x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# draw a image
# image_0 = x_train[0]
# plt.imshow(image_0, cmap='Greys_r')
# plt.show()

X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input")
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
XX = tf.reshape(X, [-1, 784])

Y = tf.nn.softmax(tf.matmul(XX, W) + b, name="output")
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# todo: here auto update W, b, there are some bugs in your codes
# train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    batch_count = int(mnist.train.num_examples / batch_size)
    for epoch in range(training_epochs):
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape(-1, 28, 28, 1)
            batch_y = dense_y(batch_y)
            sess.run([train_step, summary_op], feed_dict={X: batch_x, Y_: batch_y})
        print("Epoch: ", epoch)
        x_test = mnist.test.images
        x_test = x_test.reshape(-1, 28, 28, 1)
        print("Accuracy: ", accuracy.eval(feed_dict={X: x_test, Y_: dense_y(mnist.test.labels)}))
    print("done")

    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num].reshape(28, 28, 1)

    classification = sess.run(tf.argmax(Y, 1), feed_dict={X: [img]})
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num]))

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./../model/softmax_mnist_classify/saved_mnist_cnn.ckpt")
    print("Model saved to %s" % save_path)
