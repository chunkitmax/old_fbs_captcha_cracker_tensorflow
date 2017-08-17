import os.path
import tensorflow as tf
import numpy as np
import cv2
from time import time

CHAR_LIST = [ 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'r', 's', 't', 'y', 'x' ]

TARGET_MODEL_PATH = "model/model9400"

CHAR_COUNT = 6
CLASS_COUNT = 13

class Recognizer:
    
    def __init__(self, model_path = TARGET_MODEL_PATH):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.import_meta_graph(model_path + '.meta')
        self.saver.restore(self.sess, model_path)
        self.x = self.sess.graph.get_tensor_by_name('x:0')
        # y_ = self.sess.graph.get_tensor_by_name('y_:0')
        self.is_training = self.sess.graph.get_tensor_by_name('is_training:0')
        self.predict = []
        for i in range(6):
            self.predict.append(self.sess.graph.get_tensor_by_name('predict' + str(i) + ':0'))
        # accuracy = self.sess.graph.get_tensor_by_name('Mean:0')

    def __del__(self):
        self.sess.close()
        
    def recognize(self, img):
        img_dims = len(np.shape(img))
        if img_dims < 3:
            raise ValueError("Image must be at least rank 3")
        elif img_dims == 3:
            img = [img]
        prediction = self.sess.run(self.predict, feed_dict={ self.x: img, self.is_training: False })
        results = "".join([CHAR_LIST[int(i)] for i in prediction])
        return results

    def readImage(self, path):
        image = cv2.imread(path, 1)
        return [image]
