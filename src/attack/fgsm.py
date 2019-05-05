import tensorflow as tf
import numpy as np


class FGSM_attack:
    def __init__(self, model,  batch_shape,  max_epsilon,  img_bounds=[-1, 1], n_classes=10):
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        self.y_input = tf.placeholder(tf.int32, shape=(batch_shape[0]))
        
        y_onehot = tf.one_hot(self.y_input, n_classes)
        logits = model(self.x_input)
        logits_correct_class = tf.reduce_sum(logits * y_onehot, axis=1)

        self.loss = tf.reduce_mean(logits_correct_class)
        self.grad = tf.gradients(self.loss, self.x_input)

        self.max_epsilon = max_epsilon
        self.batch_shape = batch_shape
        self.img_bounds = img_bounds
        
    def generate(self, sess, images, labels_or_targets, verbose=False):
        delta_init = np.zeros(np.shape(images), dtype=np.float32)  
            
        delta = delta_init

        lower_bounds = np.maximum(self.img_bounds[0] - images, -self.max_epsilon)
        upper_bounds = np.minimum(self.img_bounds[1] - images, self.max_epsilon)
        
        l, gradients  = sess.run([self.loss, self.grad], feed_dict={self.x_input:images + delta, self.y_input:labels_or_targets})
        
        delta = delta - gradients[0]
        delta = np.clip(delta, lower_bounds, upper_bounds)

        return images + delta