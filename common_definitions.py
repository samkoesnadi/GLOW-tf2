import tensorflow as tf
import math
import numpy as np
tf.random.set_seed(42)

# model parameter
SQUEEZE_FACTOR = 4
K_GLOW = 4
L_GLOW = 1
ACTIVATION = tf.nn.elu
KERNEL_INITIALIZER_CLOSE_ZERO = tf.random_normal_initializer(0, 1e-5)
# KERNEL_INITIALIZER_CLOSE_ZERO = tf.keras.initializers.he_normal()
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()
HARD_KERNEL_REGULARIZER = tf.keras.regularizers.L1L2(l1=0.5, l2=0.5)
SOFT_KERNEL_REGULARIZER = tf.keras.regularizers.l2(0.3)
NUM_CLASSES = 100

# training parameters
LEARNING_RATE = 1e-4
REGULARIZER_N = 1e-3
DROPOUT_N = 0.05
BATCH_SIZE = 256
SHUFFLE_SIZE = 10000
EPOCHS = 100000
IMG_SIZE = 28  # better to be mult of SQUEEZE_FACTOR
CHANNEL_SIZE = 1


# general
TF_EPS = tf.keras.backend.epsilon()