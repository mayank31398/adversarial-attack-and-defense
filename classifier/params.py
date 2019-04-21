import os

######################################### DEVNAGRI #########################################
WEIGHTS_SAVE_PATH = "Training_weights_DEVNAGRI"
IMAGE_SHAPE = (28, 28, 1)
CLASSES = 10
BATCH_SIZE = 64
ITERATIONS = 100000
SAVE_WEIGHTS_EVERY = 10000
TEST_EVERY = 200
CODE_FILES = ["params.py", "train.py", "DEVNAGRI.py", "utils.py"]

CLASSIFIER_LEARNING_RATE = 1e-3
CLASSIFIER_HALF_LIFE = 5000
CLASSIFIER_DECAY_RATE = 0.5
CLASSIFIER_STAIRCASE = True
