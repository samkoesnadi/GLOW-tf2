from common_definitions import *
from utils.utils import *


class InvConv1(tf.keras.layers.Layer):
    """
    This is replacement of fixed permutation
    The weight has to be guaranteed to be square-sized, no bias
    """

    def __init__(self, channel_size):
        super().__init__()
        self.W = self.add_weight("W", shape=(channel_size, channel_size),
                                 initializer=tf.keras.initializers.Orthogonal(),
                                 regularizer=det_1_reg,
                                 trainable=True)

    def call(self, inputs, logdet=False, reverse=False):
        W = self.W

        if reverse:
            x = tf.einsum("ml,ijkl->ijkm", tf.linalg.inv(W), inputs)
        else:
            x = tf.einsum("ml,ijkl->ijkm", W, inputs)

        if logdet:
            return x, inputs.shape[1] * inputs.shape[2] * tf.math.log(tf.math.abs(tf.linalg.det(W)))
        else:
            return x, None


class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        # !!! Do not change the order of the add weight
        self.bn = tf.keras.layers.BatchNormalization()

        # # temp var
        # self._first = True

    def call(self, inputs, logdet=False, reverse=False, training=False):
        if reverse:
            beta = self.bn.beta
            gamma = self.bn.gamma
            variance = self.bn.moving_variance
            mean = self.bn.moving_mean
            epsilon = self.bn.epsilon

            x = (inputs - beta) / gamma
            x = x * tf.math.sqrt(variance + epsilon) + mean
        else:
            x = self.bn(inputs, training=training)

        if logdet:
            variance = self.bn.moving_variance
            epsilon = self.bn.epsilon
            gamma = self.bn.gamma
            return x, inputs.shape[1] * inputs.shape[2] * tf.math.log(
                tf.math.abs(tf.math.reduce_prod(gamma * (variance + epsilon) ** (-.5))))
        else:
            return x, None


class AffineCouplingLayer(tf.keras.layers.Layer):
    def __init__(self, channel_size, no_act=False):
        """

        :param channel_size:
        :param no_act: no activation in forward and backward (important for last layer)
        """
        super().__init__()

        # variables
        self.channel_size = channel_size
        self.no_act = no_act

        self.nn1 = self.nnLayer(channel_size)
        self.nn2 = self.nnLayer(channel_size)

        # self.W = self.add_weight("W", shape=(1, 1, 1, channel_size,),
        #                          initializer=KERNEL_INITIALIZER,
        #                          trainable=True)

    def nnLayer(self, channel_size):
        inputs = tf.keras.Input(shape=(None, None, channel_size // 2))

        x = tf.keras.layers.Conv2D(512, 3, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(512, 1, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        s = tf.keras.layers.Conv2D(channel_size // 2, 3, kernel_initializer=KERNEL_INITIALIZER_CLOSE_ZERO, padding="same")(x)
        t = tf.keras.layers.Conv2D(channel_size // 2, 3, kernel_initializer=KERNEL_INITIALIZER_CLOSE_ZERO, padding="same")(x)

        return tf.keras.Model(inputs, [tf.exp(s), t])

    def forward_block(self, x, s, t):
        y = x * s + t
        y = y if self.no_act else tf.nn.tanh(y)
        return y

    def backward_block(self, y, s, t):
        x = ((y if self.no_act else tf.math.atanh(y)) - t) / s
        return x

    def call(self, inputs, logdet=False, reverse=False, training=False):
        if reverse:
            v1, v2 = split_last_channel(inputs)
            s1, t1 = self.nn1(v1, training=training)
            u2 = self.backward_block(v2, s1, t1)
            s2, t2 = self.nn2(u2, training=training)
            u1 = self.backward_block(v1, s2, t2)

            # change convention for variable purpose
            v1 = u1
            v2 = u2
            # print(v1[0, 0, 0, 0], v2[0, 0, 0, 0])
        else:
            u1, u2 = split_last_channel(inputs)
            s2, t2 = self.nn2(u2, training=training)
            v1 = self.forward_block(u1, s2, t2)
            s1, t1 = self.nn1(v1, training=training)
            v2 = self.forward_block(u2, s1, t1)
            # print(u1[0,0,0,0], u2[0,0,0,0])

        if logdet:
            _logabsdet = tf.reduce_mean(log_abs(s1) + log_abs(s2), 0)
            return (v1, v2), tf.reduce_sum(_logabsdet)
        else:
            return (v1, v2), None


class FlowStep(tf.keras.layers.Layer):
    def __init__(self, channel_size, no_act=False):
        super().__init__()
        self.bn = BatchNormalization()
        self.perm = InvConv1(channel_size)
        self.acl = AffineCouplingLayer(channel_size, no_act)

    def call(self, inputs, logdet=False, reverse=False, training=False):
        if not reverse:
            # act norm
            x, logdet_an = self.bn(inputs, logdet, reverse, training)

            # invertible 1x1 layer
            x, logdet_perm = self.perm(x, logdet, reverse)

            # affine coupling layer
            x, logdet_acl = self.acl(x, logdet, reverse, training)
        else:
            # affine coupling layer
            x, _ = self.acl(inputs, logdet, reverse, training)
            x = tf.concat(x, axis=-1)  # concat the two output produced

            # invertible 1x1 layer
            x, _ = self.perm(x, logdet, reverse)

            # act norm
            x, _ = self.bn(x, logdet, reverse, training)

        if logdet:
            # print(logdet_an, logdet_perm, logdet_acl)
            return x, logdet_an + logdet_perm + logdet_acl
        else:
            return x, None


class CropIfNotFitLayer(tf.keras.layers.Layer):
    def __init__(self, factor_size):
        super().__init__()
        self.factor_size = factor_size

    def call(self, inputs):
        shape = inputs.get_shape()
        height = int(shape[1])
        width = int(shape[2])
        if height % self.factor_size == 0 and width % self.factor_size == 0:
            x = inputs
        else:
            x = tf.image.crop_to_bounding_box(
                inputs, 0, 0, height // self.factor_size * self.factor_size,
                              width // self.factor_size * self.factor_size
            )
        return x


class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, factor_size):
        super().__init__()
        self.factor_size = factor_size

    def build(self, input_shape):
        self._input_shape = input_shape

    def call(self, inputs, reverse=False):
        if reverse:
            return unsqueeze2d(inputs, self.factor_size)
        else:
            return squeeze2d(inputs, self.factor_size)


class GLOW(tf.keras.Model):
    def __init__(self, factor_size, K, L):
        super().__init__()

        # variables
        sqrt_factor_size = int(factor_size ** .5)  # sqrt the factor size as it is per dimension
        self.channel_order = [int(CHANNEL_SIZE * factor_size ** (l + 1) / 2 ** l) for l in
                              range(L)]  # channel order in the multi-scale architecture

        # layers
        # self.cropifnotfitlayer = CropIfNotFitLayer(sqrt_factor_size)
        self.squeezelayers = [SqueezeLayer(sqrt_factor_size) for _ in range(L)]
        self.flowsteps = [[FlowStep(c, no_act=((i_l == L - 1) and (k == K - 1))) for k in range(K)] for i_l, c in
                          enumerate(self.channel_order)]
        self.mean_logsd_nns = [
            tf.keras.layers.Dense(c, trainable=False, kernel_initializer=tf.keras.initializers.zeros())
            for c in self.channel_order]  # TODO: this might need to be configured

        # constant var
        self.factor_size = factor_size
        self.K = K
        self.L = L

        # self.debugging = {}

    # @tf.function
    def call(self, inputs, logdet=False, reverse=False, training=False):
        inputs = tf.cast(inputs, dtype=tf.float32)  # cast it

        if not reverse:
            # # # crop to matches the squeeze function
            # # x = self.cropifnotfitlayer(inputs)
            # x = tf.clip_by_value(inputs, 1e-3, 1 - 1e-3)

            # # Step 1. first invert sigmoid assuming input is [0..1]
            # x = inv_sigmoid(x)
            x = inputs

            # run inner iteration of L-1 times
            z = []
            logdet_fs_total = 0

            # Step 2.
            for i_l in range(self.L):
                # Step 2.1
                x = self.squeezelayers[i_l](x, reverse)

                # Step 2.2 run flow step for K times
                for i_k in range(self.K - 1):
                    (ya, yb), logdet_fs = self.flowsteps[i_l][i_k](x, logdet, reverse, training)
                    x = concat_last_channel(yb, ya)  # flip the ya and yb as of the architecture design
                    if logdet: logdet_fs_total += logdet_fs

                # Step 2.3 run the last K without concat
                (ya, yb), logdet_fs = self.flowsteps[i_l][self.K - 1](x, logdet, reverse, training)
                if logdet: logdet_fs_total += logdet_fs

                # set x to yb
                x = yb
                ya, yb = tf.reshape(ya, [ya.shape[0], -1]), tf.reshape(yb, [yb.shape[0], -1])

                # append ya to z
                z.append(tf.reshape(ya, [ya.shape[0], -1]))

                if i_l == self.L - 1:
                    z.append(tf.reshape(yb, [yb.shape[0], -1]))

            z_total = tf.concat(z, axis=-1)
            # print(logpzs, logdet_fs_total)
            if logdet:
                # this is when the objective is defined
                mean_var = tf.zeros(z_total.shape)
                mean = mean_var
                var = mean_var + 1
                logpzs = logpz(mean, var, z_total)

                # print(logpzs, logdet_fs_total)
                # print(logpzs / BATCH_SIZE, logdet_fs_total)
                return z_total, logpzs / BATCH_SIZE + logdet_fs_total
            else:
                return z_total, None
        else:
            assert not logdet  # inv cant have logdet
            z_total = inputs
            # print(tf.reduce_max(inputs))
            z_shape = z_total.get_shape()
            z_sizes = [int(z_shape[1] / 2 ** (i_l + 1)) for i_l in
                       range(self.L)]  # the sizes as effect to the multi-scale arch
            x = None

            for i_l, z_size in enumerate(z_sizes[::-1]):
                if i_l == 0:
                    z_size *= 2
                i_l = self.L - i_l - 1  # reverse the index

                z_total, z = split_last_channel(z_total, boundary=-z_size)  # get the z

                za_channel_size = self.channel_order[i_l] if i_l == self.L - 1 else self.channel_order[i_l] // 2
                wh_size = int((z_size / za_channel_size) ** .5)

                if i_l == self.L - 1:
                    z1, z2 = split_last_channel(z)
                    z1 = tf.reshape(z1, [-1, wh_size, wh_size, za_channel_size // 2])
                    z2 = tf.reshape(z2, [-1, wh_size, wh_size, za_channel_size // 2])
                    z = concat_last_channel(z1, z2)
                else:
                    z = tf.reshape(z, [-1, wh_size, wh_size, za_channel_size])
                    z = concat_last_channel(z, x)  # concat the z and previous x

                # run the last K
                # self.debugging[f"flow_{i_l}_{self.K-1}"].append(z[0,0,0,0].numpy())
                x, _ = self.flowsteps[i_l][self.K - 1](z, logdet, reverse)

                # run flow step for K times
                for i_k in reversed(range(self.K - 1)):
                    # self.debugging[f"flow_{i_l}_{i_k}"].append(x[0,0,0,0].numpy())
                    # unswitch the two layers
                    xa, xb = split_last_channel(x)
                    x = concat_last_channel(xb, xa)

                    x, _ = self.flowsteps[i_l][i_k](x, logdet, reverse)

                # unsqueeze
                # self.debugging[f"squeeze_{i_l}"].append(x[0,0,0,0].numpy())
                x = self.squeezelayers[i_l](x, reverse)

            # # Step 1. sigmoid image
            # x = tf.nn.sigmoid(x)

            return x, None


if __name__ == "__main__":
    a = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, CHANNEL_SIZE), 0, 1)

    a = my_tf_round(a, 3)  # round it

    # x = InvConv1(32)
    # print(x(x(a, inverse_mode=False), inverse_mode=True))

    # x = FlowStep(32)
    # print(x(a))

    model = GLOW(SQUEEZE_FACTOR, K_GLOW, L_GLOW)
    import time
    # model.load_weights(CHECKPOINT_PATH+".h5")

    # print(model(a, logdet=True, reverse=False))
    a_1 = model(model(a)[0], reverse=True)[0]
    a_1 = my_tf_round(a_1, 3)  # round it

    print(tf.reduce_sum(tf.cast(a!=a_1, dtype=tf.float32)))

    tf.assert_equal(a_1, a)

    # import csv
    # with open('mycsvfile.csv', 'w') as f:  # Just use 'w' mode in 3.x
    #     w = csv.DictWriter(f, model.debugging.keys())
    #     w.writeheader()
    #     w.writerow(model.debugging)
