import tensorflow as tf
import math
import numpy as np
tf.random.set_seed(42)

# model parameter
SQUEEZE_FACTOR = 4
K_GLOW = 16
L_GLOW = 2
ACTIVATION = tf.nn.relu
KERNEL_INITIALIZER_CLOSE_VALUE = lambda x=0: tf.random_normal_initializer(x, 1e-4)
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()
KERNEL_REGULARIZER = tf.keras.regularizers.l2()

# training parameters
LEARNING_RATE = 1e-3
REGULARIZER_N = 0.
# LAMBDA_LIPSCHITZ = 1e-3
BATCH_SIZE = 128
SHUFFLE_SIZE = 10000
EPOCHS = 100000
IMG_SIZE = 28  # better to be mult of SQUEEZE_FACTOR
CHANNEL_SIZE = 1
CHECKPOINT_PATH = "./checkpoints/weights"
TENSORBOARD_LOGDIR = "./logs/GLOW"

# dataset parameters
ALPHA_BOUNDARY = 0.05

# general
TF_EPS = tf.keras.backend.epsilon()