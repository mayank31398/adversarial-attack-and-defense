import scipy.io as sio
import numpy as np
import keras
from models.vae import vae_model_svhn, vae_model_fmnist
from loaddata import load_fmnist
from keras import backend as K
import tensorflow as tf
from data.data import load_svhn
from jpeg import rgb_to_ycbcr

X_train, Y_train, X_test, Y_test, labels_train, labels_test = load_fmnist()

# Load dataset
train_x, train_y, train_l = load_svhn()
test_x, test_y, test_l = load_svhn("test")

# Reshape
train_x = train_x.reshape([-1, 32, 32, 3])
test_x = test_x.reshape([-1, 32, 32, 3])

sess = tf.Session()
keras.backend.set_session(sess)

vae_svhn = vae_model_svhn()
vae_fmnist = vae_model_fmnist()

vae_svhn.compile(optimizer='adam')
vae_fmnist.compile(optimizer='adam')

vae_svhn.fit(train_x, epochs=200, batch_size=64, verbose=2)

vae_fmnist.fit(X_train, epochs=200, batch_size=64, verbose=2)