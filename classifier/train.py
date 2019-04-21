import os
from datetime import datetime

import numpy as np
from comet_ml import Experiment
import tensorflow as tf
from sklearn.metrics import accuracy_score

from DEVNAGRI import Classifier
from params import (BATCH_SIZE, CLASSES, CLASSIFIER_DECAY_RATE,
                    CLASSIFIER_HALF_LIFE, CLASSIFIER_LEARNING_RATE,
                    CLASSIFIER_STAIRCASE, IMAGE_SHAPE, ITERATIONS,
                    SAVE_WEIGHTS_EVERY, TEST_EVERY, WEIGHTS_SAVE_PATH)
from utils import (DEVNAGRIDataLoader, DisplayTest, DisplayTrain, MakeNotSoHot,
                   MakeOneHot)

BATCH_SIZE = 128
EPOCHS = 101
EXPERIMENT = Experiment(
    project_name="adversarial",
    workspace="mayank31398",
    auto_output_logging=None,
    auto_metric_logging=False,
    auto_param_logging=False
)


def GetOptimizer(
    learning_rate,
    global_step,
    decay_step,
    decay_rate,
    staircase,
    loss,
    trainable_variables
):
    scheduler = tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_step,
        decay_rate,
        staircase=staircase
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=scheduler).minimize(
        loss,
        var_list=trainable_variables,
        global_step=global_step
    )

    return optimizer


def Train(dataloader, classifier: Classifier):
    devnagri_train_loader = dataloader.GetTrainLoader()
    devnagri_test_loader = dataloader.GetTestLoader()

    x = tf.placeholder(tf.float32, shape=[
                       None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]])
    y = tf.placeholder(tf.float32, shape=[None, CLASSES])

    predictions = classifier.Classify(x)
    classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=predictions))

    # Classifier optimizer
    global_step = tf.Variable(0, trainable=False)
    classifier_optimizer = GetOptimizer(
        CLASSIFIER_LEARNING_RATE,
        global_step,
        CLASSIFIER_HALF_LIFE,
        CLASSIFIER_DECAY_RATE,
        CLASSIFIER_STAIRCASE,
        classification_loss,
        tf.trainable_variables("Encoder") + tf.trainable_variables("Decoder")
    )

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for iteration in range(ITERATIONS):
            mini_batch_x, mini_batch_y = next(devnagri_train_loader)

            _, classification_loss_value, prediction_values = sess.run(
                [classifier_optimizer, classification_loss, predictions], feed_dict={x: mini_batch_x, y: MakeOneHot(mini_batch_y)})

            classification_accuracy = accuracy_score(
                mini_batch_y, MakeNotSoHot(prediction_values))

            DisplayTrain(iteration, classification_loss_value,
                         classification_accuracy)

            if(iteration % TEST_EVERY == 0):
                predictions = []
                truths = []
                classification_loss_value = 0

                for _ in range(dataloader.test_batches):
                    mini_batch_x, mini_batch_y = next(devnagri_test_loader)

                    classification_loss_value_, prediction_values = sess.run(
                        [classification_loss, predictions], feed_dict={x: mini_batch_x, y: MakeOneHot(mini_batch_y)})
                    classification_loss_value += classification_loss_value_

                    truths.append(mini_batch_y)
                    predictions.append(MakeNotSoHot(prediction_values))

                truths = np.concatenate(truths)
                predictions = np.concatenate(predictions)

                classification_accuracy = accuracy_score(truths, predictions)

                DisplayTest(classification_loss_value,
                            classification_accuracy)

            if(iteration % SAVE_WEIGHTS_EVERY == 0):
                os.makedirs(os.path.join(WEIGHTS_SAVE_PATH,
                                         "checkpoints"), exist_ok=True)

                saver.save(sess, os.path.join(WEIGHTS_SAVE_PATH,
                                              "checkpoints", str(iteration) + ".ckpt"))


def main():
    dataloader = DEVNAGRIDataLoader(BATCH_SIZE)
    classifier = Classifier()
    Train(dataloader, classifier)


if(__name__ == "__main__"):
    main()
