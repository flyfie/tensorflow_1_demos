from scipy import misc
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


class TestResult:
    def __init__(self):
        self.anger = 0
        self.disgust = 0
        self.fear = 0
        self.happy = 0
        self.sad = 0
        self.surprise = 0
        self.neutral = 0

    def evaluate(self, label):
        if 0 == label:
            self.anger = self.anger + 1
        if 1 == label:
            self.disgust = self.disgust + 1
        if 2 == label:
            self.fear = self.fear + 1
        if 3 == label:
            self.happy = self.happy + 1
        if 4 == label:
            self.sad = self.sad + 1
        if 5 == label:
            self.surprise = self.surprise + 1
        if 6 == label:
            self.neutral = self.neutral + 1

    def display_result(self, evaluations):
        print("anger = " + str((self.anger / float(evaluations)) * 100) + "%")
        print("disgust = " + str((self.disgust / float(evaluations)) * 100) + "%")
        print("fear = " + str((self.fear / float(evaluations)) * 100) + "%")
        print("happy = " + str((self.happy / float(evaluations)) * 100) + "%")
        print("sad = " + str((self.sad / float(evaluations)) * 100) + "%")
        print("surprise = " + str((self.surprise / float(evaluations)) * 100) + "%")
        print("neutral = " + str((self.neutral / float(evaluations)) * 100) + "%")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


img = mpimg.imread('./data/emotion_detection/author_image.jpg')
gray = rgb2gray(img)
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.show()

sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('../model/EmotionDetector_logs/model.ckpt-1000.meta')
new_saver.restore(sess, '../model/EmotionDetector_logs/model.ckpt-1000')
tf.get_default_graph().as_graph_def()
x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")


image_test = np.resize(gray, (1, 48, 48, 1))
tResult = TestResult()
num_evaluations = 1000
for i in range(0, num_evaluations):
    result = sess.run(y_conv, feed_dict={x: image_test})
    label = sess.run(tf.argmax(result, 1))
    label = label[0]
    label = int(label)
    tResult.evaluate(label)
tResult.display_result(num_evaluations)

