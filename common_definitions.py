import tensorflow as tf
import math
import numpy as np
tf.random.set_seed(42)

DATASET = "mnist"  # dataset to train on

# model parameter
SQUEEZE_FACTOR = 4
K_GLOW = 16
L_GLOW = 2
IMG_SIZE = 28  # better to be mult of SQUEEZE_FACTOR
CHANNEL_SIZE = 1
ACTIVATION = tf.nn.relu
KERNEL_INITIALIZER_CLOSE_VALUE = lambda x=0: tf.random_normal_initializer(x, 1e-4)
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

# training parameters
LEARNING_RATE = 1e-3
# LAMBDA_LIPSCHITZ = 1e-3
DROPOUT_N = 0.1
BATCH_SIZE = 128
SHUFFLE_SIZE = 10000
EPOCHS = 1000
CHECKPOINT_PATH = "./checkpoints/weights"
TENSORBOARD_LOGDIR = "./logs/GLOW"

# dataset parameters
ALPHA_BOUNDARY = 0.05

# general
TF_EPS = tf.keras.backend.epsilon()