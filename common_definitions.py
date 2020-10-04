import tensorflow as tf
import math
import numpy as np
tf.random.set_seed(42)

# model parameter
SQUEEZE_FACTOR = 4
K_GLOW = 8
L_GLOW = 3
ACTIVATION = tf.nn.elu
ALPHA_LEAKY_RELU = 0.5  # for activation in forward/backward block not the nn
KERNEL_INITIALIZER_CLOSE_ZERO = tf.random_normal_initializer(0, 1e-5)
# KERNEL_INITIALIZER_CLOSE_ZERO = tf.keras.initializers.he_normal()
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()
# HARD_KERNEL_REGULARIZER = tf.keras.regularizers.L1L2(l1=0.03, l2=0.03)
# SOFT_KERNEL_REGULARIZER = tf.keras.regularizers.l2(0.01)
NUM_CLASSES = 100

# training parameters
LEARNING_RATE = 1e-3
REGULARIZER_N = 1e-3
LAMBDA_LIPSCHITZ = 1e-3
BATCH_SIZE = 5
SHUFFLE_SIZE = 10000
EPOCHS = 100000
IMG_SIZE = 32  # better to be mult of SQUEEZE_FACTOR
CHANNEL_SIZE = 3
CHECKPOINT_PATH = "./checkpoints/weights"
TENSORBOARD_LOGDIR = "./logs/GLOW"

# general
TF_EPS = tf.keras.backend.epsilon()