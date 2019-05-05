import tensorflow as tf
import numpy as np


class PGD_attack:
    def __init__(self, model,  batch_shape,  max_epsilon,  max_iter,  targeted,  img_bounds=[-1, 1], use_noise=True, initial_lr=0.5,  lr_decay=0.98, n_classes=10, rng = np.random.RandomState()):
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        self.y_input = tf.placeholder(tf.int32, shape=(batch_shape[0]))
        
        # Loss function: the mean of the logits of the correct class
        y_onehot = tf.one_hot(self.y_input, n_classes)
        logits = model(self.x_input)
        logits_correct_class = tf.reduce_sum(logits * y_onehot, axis=1)

        self.loss = tf.reduce_mean(logits_correct_class)
        self.grad = tf.gradients(self.loss, self.x_input)

        # Keep track of the parameters:
        self.targeted = targeted
        self.max_iter = max_iter
        self.max_epsilon = max_epsilon
        self.batch_shape = batch_shape
        self.img_bounds = img_bounds
        self.use_noise = use_noise
        self.rng = rng
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        
    def generate(self, sess, images, labels_or_targets, verbose=False):
        if self.use_noise:
            alpha = self.max_epsilon * 0.5
            delta_init = alpha * np.sign(self.rng.normal(size=np.shape(images))).astype(np.float32)
        else:
            delta_init = np.zeros(np.shape(images), dtype=np.float32)
            

        lr = self.initial_lr
        delta = delta_init
        
        if self.targeted:
            multiplier = 1.
        else:
            multiplier = -1.

        lower_bounds = np.maximum(self.img_bounds[0] - images, -self.max_epsilon)
        upper_bounds = np.minimum(self.img_bounds[1] - images, self.max_epsilon)
        
       
        for i in range(self.max_iter):
            l, gradients  = sess.run([self.loss, self.grad], 
                                 feed_dict={self.x_input:images + delta,
                                            self.y_input:labels_or_targets})
            
            delta = delta + multiplier * lr * gradients[0]
            delta = np.clip(delta, lower_bounds, upper_bounds)
            
            lr = lr * self.lr_decay
            
            if verbose:
                print('Iter %d, loss: %.2f' % (i, l))
        return images + delta