from common_definitions import *
from model import *
import os


class Brain:
    def __init__(self, factor_size, k, l, learning_rate=LEARNING_RATE):
        self.model = GLOW(factor_size, k, l)

        # vars for training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=.5, amsgrad=True)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z, logpx = self.model(inputs, logdet=True)

            # define the negative log-likelihood
            nll = -logpx
            nll_and_reg = nll + REGULARIZER_N * tf.add_n(self.model.losses)
        # print(tf.add_n(self.model.losses))
        model_gradients = tape.gradient(nll_and_reg, self.model.trainable_variables)
        # tf.print([tf.reduce_mean(tf.abs(m_g)) for m_g in model_gradients])
        self.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))

        return nll

    def save_weights(self, path):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.save_weights(path+"an.h5")
        self.critic_network.save_weights(path+"cn.h5")
        self.critic_target.save_weights(path+"ct.h5")
        self.actor_target.save_weights(path+"at.h5")

    def load_weights(self, path):
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_network.load_weights(path + "cn.h5")
            self.critic_target.load_weights(path + "ct.h5")
            self.actor_target.load_weights(path + "at.h5")
        except:
            return "Weights cannot be loaded"
        return "Weights loaded"