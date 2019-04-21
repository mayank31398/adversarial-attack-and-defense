import tensorflow as tf


class Classifier:
    def Classify(self, x):
        with tf.variable_scope("Encoder"):
            # 28, 28, 1
            x = tf.layers.conv2d(x, kernel_size=5, filters=4,
                                 activation=tf.nn.leaky_relu)  # 24, 24, 4
            
            x = tf.layers.conv2d(x, kernel_size=5, filters=16,
                                 activation=tf.nn.leaky_relu)  # 20, 20, 16

            x = tf.layers.conv2d(x, kernel_size=5, filters=16,
                                 activation=tf.nn.leaky_relu)  # 16, 16, 16
            
            x = tf.layers.conv2d(x, kernel_size=5, filters=16,
                                 activation=tf.nn.leaky_relu)  # 12, 12, 16
            
            x = tf.layers.conv2d(x, kernel_size=5, filters=32,
                                 activation=tf.nn.leaky_relu)  # 8, 8, 32

            x = tf.layers.conv2d(x, kernel_size=5, filters=32,
                                 activation=tf.nn.leaky_relu)  # 4, 4, 32

            x = tf.layers.conv2d(x, kernel_size=3, filters=32,
                                 activation=tf.nn.leaky_relu)  # 2, 2, 32

            x = tf.layers.flatten(x)  # 128
            # LATENT_DIMENSION
            x = tf.layers.dense(x, 64, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 16, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, 10, activation=tf.nn.softmax)

        return x
