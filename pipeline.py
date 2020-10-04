from common_definitions import *
from model import *
import os


class Brain:
    def __init__(self, factor_size, k, l, learning_rate=LEARNING_RATE):
        self.model = GLOW(factor_size, k, l)

        # vars for training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape_inside:
                tape_inside.watch(inputs)
                z, logpx = self.model(inputs, logdet=True)

                # define the negative log-likelihood
                nll = -logpx
                nll_and_reg = nll + REGULARIZER_N * tf.add_n(self.model.losses)
            gradient_to_inputs = (tf.norm(tape_inside.gradient(nll, inputs)) - 1) ** 2
            nll_and_reg += LAMBDA_LIPSCHITZ * gradient_to_inputs

        # print(tf.add_n(self.model.losses))
        model_gradients = tape.gradient(nll_and_reg, self.model.trainable_variables)
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
            self.actor_network.load_weights(path + ".h5")
        except:
            return "Weights cannot be loaded"
        return "Weights loaded"