from common_definitions import *
from model import *
import os


class Brain:
    def __init__(self, factor_size, k, l, img_size, channel_size, learning_rate=LEARNING_RATE):
        self.model = GLOW(factor_size, k, l, img_size, channel_size)

        # # lr scheduler
        # lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        #     learning_rate,
        #     decay_steps=1111,
        #     decay_rate=0.91,
        #     staircase=True)

        # vars for training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def sample(self, temp=1.):
        return self.model.sample(temp)

    def forward(self, inputs):
        return self.model(inputs)[0]

    def backward(self, inputs):
        return self.model(inputs, reverse=True)[0]

    @tf.function
    def train_step(self, inputs):
        # with tf.GradientTape() as tape:
        #     with tf.GradientTape() as tape_inside:
        #         tape_inside.watch(inputs)
        #         z, logpx = self.model(inputs, logdet=True, training=True)
        #
        #         # define the negative log-likelihood
        #         nll = tf.clip_by_value(-logpx, -1e9, 1e9)
        #     gradient_to_inputs = (tf.norm(tape_inside.gradient(nll, inputs)) - 1) ** 2
        #     nll += LAMBDA_LIPSCHITZ * gradient_to_inputs

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            z, logpx = self.model(inputs, logdet=True, training=True)

            # define the negative log-likelihood
            nll = tf.clip_by_value(-logpx, -1e9, 1e9)

        model_gradients = tape.gradient(nll, self.model.trainable_variables)
        # tf.print([tf.reduce_mean(tf.abs(m_g)) for m_g in model_gradients])
        self.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))

        return z, nll

    def save_weights(self, path):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.model.save_weights(path+".h5")

    def load_weights(self, path):
        try:
            self.model.load_weights(path + ".h5")
        except:
            return "Weights cannot be loaded"
        return "Weights loaded"