from loaddata import load_svhn, load_fmnist
from models.fmnistmodel import fmnist_model
from models.svhnmodel import svhn_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler
import numpy as np
import keras

X_train, Y_train, X_test, Y_test, labels_train, labels_test = load_fmnist()

train_x, train_y, train_l = load_svhn()
test_x, test_y, test_l = load_svhn("test")

train_x = train_x.reshape([-1, 32, 32, 3])
test_x = test_x.reshape([-1, 32, 32, 3])

sess = tf.Session()
keras.backend.set_session(sess)

fmnist_classifier = fmnist_model()
svhn_classifier = svhn_model()

optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=False)
svhn_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
fmnist_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

fmnist_classifier.fit(X_train, Y_train, batch_size=256, epochs=150, validation_data=(X_test, Y_test), verbose=2)

svhn_classifier.fit(train_x, train_y, batch_size=256, epochs=150, validation_data=(test_x, test_y), verbose=2)
