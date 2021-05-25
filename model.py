from common_definitions import *
from utils.utils import *


class Z_Norm_IntermediateLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        channel_size = input_shape[-1]
        # self.mean_lstd = tf.keras.layers.Dense(channel_size * 2, kernel_initializer=KERNEL_INITIALIZER_CLOSE_ZERO, kernel_regularizer=KERNEL_REGULARIZER)
        self.mean_lstd = tf.keras.layers.Conv2D(channel_size * 2, 1, padding="same", kernel_initializer=KERNEL_INITIALIZER_CLOSE_VALUE(0))
        self.channel_size = channel_size
        self.img_width = input_shape[1]

    def call(self, v1, v2, logdet=False, reverse=False):
        """
        to sample forward: norm v1 with mean and lvar learned from v2
        :param v1:
        :param v2:
        :return:
        """
        mean_lstd = self.mean_lstd(v2)
        mean, lstd = split_last_channel(mean_lstd)
        std = tf.exp(lstd)

        if reverse:
            output = v1 * std + mean
        else:
            output = (v1 - mean) / std

        if logdet:
            # TODO check this logdet
            return output, tf.reduce_mean(logpz(mean, lstd, v1), 0)
            # return output, tf.reduce_mean(tf.math.reciprocal_no_nan(std), 0)
        else:
            return output, 0.

    def sample(self, v2, temp=1.):
        mean_lstd = self.mean_lstd(v2)
        mean, lstd = split_last_channel(mean_lstd)
        std = tf.exp(lstd) * temp
        return mean + std * tf.random.normal((1, self.img_width, self.img_width, self.channel_size))


class Z_Norm_LastLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        channel_size = input_shape[-1]
        self.mean_lstd = self.add_weight("Mean_Logvar", (1, input_shape[1], input_shape[2], channel_size * 2,), initializer=KERNEL_INITIALIZER_CLOSE_VALUE(0), trainable=True)
        self.channel_size = channel_size
        self.img_width = input_shape[1]

    def call(self, v1, logdet=False, reverse=False):
        """
        to sample forward: norm v1 with mean and lvar learned from v2
        :param v1:
        :param v2:
        :return:
        """
        mean_lstd = self.mean_lstd
        mean, lstd = split_last_channel(mean_lstd)
        std = tf.exp(lstd)

        if reverse:
            output = v1 * std + mean
        else:
            output = (v1 - mean) / std

        if logdet:
            # TODO check this logdet
            return output, tf.reduce_mean(logpz(mean, lstd, v1), 0)
            # return output, tf.reduce_mean(tf.math.reciprocal_no_nan(std), 0)
        else:
            return output, 0.

    def sample(self, temp=1.):
        mean_lstd = self.mean_lstd
        mean, lstd = split_last_channel(mean_lstd)
        std = tf.exp(lstd) * temp
        return mean + std * tf.random.normal((1, self.img_width, self.img_width, self.channel_size))


class InvConv1(tf.keras.layers.Layer):
    """
    This is replacement of fixed permutation
    The weight has to be guaranteed to be square-sized, no bias
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        channel_size = input_shape[-1]
        self.W = self.add_weight("W_1_1", shape=(channel_size, channel_size),
                                 initializer=tf.keras.initializers.Orthogonal(),
                                 trainable=True)
        self.channel_size = channel_size

    def call(self, inputs, logdet=False, reverse=False):
        if logdet:
            if tf.linalg.det(self.W) == 0:
                W = self.W + KERNEL_INITIALIZER_CLOSE_VALUE(0)(
                    shape=self.W.shape)  # doing this will move the matrix to invertible location
            else:
                W = self.W
        else:
            W = self.W

        W = tf.reshape(tf.linalg.inv(W) if reverse else W, [1,1,self.channel_size,self.channel_size])

        x = tf.nn.conv2d(inputs, W, [1,1,1,1], padding="SAME")

        if logdet:
            return x, inputs.shape[1] * inputs.shape[2] * tf.squeeze(tf.math.log(tf.math.abs(tf.linalg.det(W)) + TF_EPS))
        else:
            return x, 0.


class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self, step_bn=100):
        super().__init__()

        self.momentum = tf.Variable(0., trainable=False)
        self._step = 0
        self.step_bn = step_bn

        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)
        # self.bn = tf.keras.layers.BatchNormalization(momentum=0.99)

    def call(self, inputs, logdet=False, reverse=False, training=False):
        # return inputs, 0
        if training and not reverse:
            if self._step < self.step_bn:
                self.momentum.assign_add(1/self.step_bn)
                self._step += 1
        if reverse:
            beta = self.bn.beta[None,None,None,:]
            gamma = self.bn.gamma[None,None,None,:]
            variance = self.bn.moving_variance[None,None,None,:]
            mean = self.bn.moving_mean[None,None,None,:]
            epsilon = self.bn.epsilon

            inv = gamma * tf.math.rsqrt(variance + epsilon)
            r_inv = tf.math.reciprocal(inv)
            x = (inputs - beta + inv * mean) * r_inv
        else:
            x = self.bn(inputs, training=training)

        if logdet:
            variance = self.bn.moving_variance
            epsilon = self.bn.epsilon
            gamma = self.bn.gamma
            return x, inputs.shape[1] * inputs.shape[2] * tf.reduce_sum(log_abs(gamma * (variance + epsilon) ** (-.5)))
        else:
            return x, 0.


class ActNormalization(tf.keras.layers.Layer):
    def __init__(self, output_only_one=False):
        super().__init__()

        # temp var
        self._initiated = False  # toggle var to initiate the value
        self.output_only_one = output_only_one

    def build(self, input_shape):
        self.channel_size = input_shape[-1]
        self.s = self.add_weight("s", shape=(1,1,1,self.channel_size), initializer=tf.keras.initializers.ones(), trainable=True)
        self.b = self.add_weight("b", shape=(1,1,1,self.channel_size), initializer=tf.keras.initializers.zeros(), trainable=True)

    def call(self, inputs, logdet=False, reverse=False):
        if (not self._initiated) and (not reverse):
            std = tf.math.reduce_std(inputs, [0,1,2])
            mean = tf.math.reduce_mean(inputs, [0,1,2])
            self.s.assign(1/std[None, None, None, :])
            self.b.assign(-mean/std[None, None, None, :])

            self._initiated = True  # change the toggle var

        if reverse:
            x = (inputs - self.b) / self.s
        else:
            x = self.s * inputs + self.b

        if logdet:
            return x, inputs.shape[1] * inputs.shape[2] * tf.reduce_sum(log_abs(self.s))
        else:
            if self.output_only_one:
                return x
            else:
                return x, 0.


class AffineCouplingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        channel_size = input_shape[-1]
        self.channel_size = channel_size

        self.nn = self.nnLayer(channel_size)


    def nnLayer(self, channel_size):
        inputs = tf.keras.Input(shape=(None, None, channel_size // 2))

        x = tf.keras.layers.Conv2D(512, 4, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, padding="same")(inputs)
        x = ActNormalization(output_only_one=True)(x)
        x = tf.keras.layers.Dropout(DROPOUT_N)(x)
        x = tf.keras.layers.Conv2D(512, 1, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, padding="same")(x)
        x = ActNormalization(output_only_one=True)(x)
        x = tf.keras.layers.Dropout(DROPOUT_N)(x)

        s = tf.keras.layers.Conv2D(channel_size // 2, 4, kernel_initializer=KERNEL_INITIALIZER_CLOSE_VALUE(2.), padding="same")(x)
        t = tf.keras.layers.Conv2D(channel_size // 2, 4, kernel_initializer=KERNEL_INITIALIZER_CLOSE_VALUE(0.), padding="same")(x)

        # postprocess s & t
        s = tf.nn.sigmoid(s)
        t = tf.nn.sigmoid(t)

        return tf.keras.Model(inputs, [s, t])

    def forward_block(self, x, s, t):
        y = x * s + t
        return y

    def backward_block(self, y, s, t):
        x = y / s - t / s
        return x

    def call(self, inputs, logdet=False, reverse=False, training=False):
        if reverse:
            v1, v2 = split_last_channel(inputs)
            s2, t2 = self.nn(v2, training=training)
            u1 = self.backward_block(v1, s2, t2)

            # change convention for variable purpose
            v1 = u1
        else:
            u1, u2 = split_last_channel(inputs)
            s2, t2 = self.nn(u2, training=training)
            v1 = self.forward_block(u1, s2, t2)
            v2 = u2

        if logdet:
            _logabsdet = tf.reduce_mean(tf.reduce_sum(log_abs(s2), [1,2,3]), 0)
            return (v1, v2), _logabsdet
        else:
            return (v1, v2), 0.


class FlowStep(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.an = ActNormalization()
        self.perm = InvConv1()
        self.acl = AffineCouplingLayer()

    def call(self, inputs, logdet=False, reverse=False, training=False):
        if not reverse:
            # act norm
            x, logdet_an = self.an(inputs, logdet, reverse)

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
            x, _ = self.an(x, logdet, reverse)

        if logdet:
            # print(logdet_an, logdet_perm, logdet_acl)
            return x, logdet_an + logdet_perm + logdet_acl
        else:
            return x, 0.


class CropIfNotFitLayer(tf.keras.layers.Layer):
    def __init__(self, factor_size):
        super().__init__()
        self.factor_size = factor_size

    def call(self, inputs, reverse=False, target_width=None):
        shape = inputs.get_shape()
        height = int(shape[1])
        width = int(shape[2])
        if reverse:
            if target_width is not None:
                _diff = target_width - width
                x = tf.pad(inputs, [[0, 0], [0, _diff], [0, _diff], [0, 0]], constant_values=0.05)
            else:
                x = inputs
        else:
            if height % self.factor_size == 0 and width % self.factor_size == 0:
                x = inputs
            else:
                x = inputs[:,:height // self.factor_size * self.factor_size,:width // self.factor_size * self.factor_size,:]
        return x


class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, factor_size):
        super().__init__()
        self.factor_size = factor_size
        self.cropLayer = CropIfNotFitLayer(factor_size=factor_size)

    def build(self, input_shape):
        self._input_shape = input_shape

    def call(self, inputs, reverse=False, target_width=None):
        if reverse:
            x = unsqueeze2d(inputs, self.factor_size)
            return self.cropLayer(x, reverse=reverse, target_width=target_width)
        else:
            cropped = self.cropLayer(inputs, reverse=reverse)
            return squeeze2d(cropped, self.factor_size)


class GLOW(tf.keras.Model):
    def __init__(self, factor_size, K, L, img_size, channel_size):
        super().__init__()

        # variables
        sqrt_factor_size = int(factor_size ** .5)  # sqrt the factor size as it is per dimension
        self.channel_size = channel_size

        # layers
        # self.cropifnotfitlayer = CropIfNotFitLayer(sqrt_factor_size)
        self.squeezelayers = [SqueezeLayer(sqrt_factor_size) for _ in range(L)]
        self.flowsteps = [[FlowStep() for _ in range(K)] for _ in range(L)]
        self.logpzlayers = [Z_Norm_IntermediateLayer() for _ in range(L-1)]
        self.logpzlayers_last = Z_Norm_LastLayer()

        # constant var
        self.factor_size = factor_size
        self.K = K
        self.L = L
        self.img_size = img_size

    def call(self, inputs, logdet=False, reverse=False, training=False):
        inputs = tf.cast(inputs, dtype=tf.float32)  # cast it

        if not reverse:
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
                    x = concat_last_channel(ya, yb)  # flip the ya and yb as of the architecture design
                    if logdet: logdet_fs_total += logdet_fs

                # Step 2.3 run the last K without concat
                (ya, yb), logdet_fs = self.flowsteps[i_l][self.K - 1](x, logdet, reverse, training)
                x = yb

                if i_l == self.L - 1:
                    x = concat_last_channel(ya, yb)
                    # logpz with the mean and var accordingly
                    ya, logpz = self.logpzlayers_last(x, logdet, reverse)
                    if logdet: logdet_fs_total += logdet_fs + logpz
                else:
                    # logpz with the mean and var accordingly
                    ya, logpz = self.logpzlayers[i_l](ya, yb, logdet, reverse)
                    if logdet: logdet_fs_total += logdet_fs + logpz

                # logpz with the mean and var accordingly
                ya = tf.compat.v1.layers.flatten(ya)

                # Step 2.4 append to the z
                z.append(ya)

            z_total = tf.concat(z, axis=-1)
            if logdet:
                return z_total, tf.squeeze(logdet_fs_total / tf.math.log(2.))  # divide by all pixel... this is now in bits/dim
            else:
                return z_total, 0.
        else:
            assert not logdet  # inv cant have logdet
            z_total = inputs
            z_sizes = [(self.img_size // 2 ** (i_l + 1))**2 * self.channel_size * 2 ** (i_l + 1) for i_l in
                       range(self.L)]  # the sizes as effect to the multi-scale arch
            x = None

            for i_l, z_size in enumerate(z_sizes[::-1]):
                if i_l == 0:
                    z_size *= 2
                i_l = self.L - i_l - 1  # reverse the index

                z_total, z = split_last_channel(z_total, boundary=-z_size)  # get the z

                channel_order = int(CHANNEL_SIZE * self.factor_size ** (i_l + 1) / 2 ** i_l)
                za_channel_size = channel_order if i_l == self.L - 1 else channel_order // 2
                wh_size = self.img_size // 2 ** (i_l + 1)

                if i_l == self.L - 1:
                    # reverse the renorm last k
                    z, _ = self.logpzlayers_last(tf.reshape(z, [-1, wh_size, wh_size, za_channel_size]), logdet, reverse)
                else:
                    # reverse the renorm last k
                    z, _ = self.logpzlayers[i_l](tf.reshape(z, [-1, wh_size, wh_size, za_channel_size]), x, logdet, reverse)

                    z = concat_last_channel(z, x)  # concat the z and previous x

                # run the last K
                x, _ = self.flowsteps[i_l][self.K - 1](z, logdet, reverse)

                # run flow step for K times
                for i_k in reversed(range(self.K - 1)):
                    x, _ = self.flowsteps[i_l][i_k](x, logdet, reverse)

                # unsqueeze
                x = self.squeezelayers[i_l](x, reverse, self.img_size // 2 ** i_l)

            return x, 0.

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def sample(self, temp=1.):
        x = None

        for i_l in reversed(range(self.L)):
            if i_l == self.L - 1:
                # reverse the renorm last k
                z = self.logpzlayers_last.sample(temp)
            else:
                # reverse the renorm last k
                z = self.logpzlayers[i_l].sample(x, temp)
                z = concat_last_channel(z, x)  # concat the z and previous x

            # run the last K
            x, _ = self.flowsteps[i_l][self.K - 1](z, reverse=True)

            # run flow step for K times
            for i_k in reversed(range(self.K - 1)):
                x, _ = self.flowsteps[i_l][i_k](x, reverse=True)

            # unsqueeze
            x = self.squeezelayers[i_l](x, reverse=True)

        return x


if __name__ == "__main__":
    a = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, CHANNEL_SIZE), 0.05, 1)

    a = my_tf_round(a, 3)  # round it

    # x = InvConv1(32)
    # print(x(x(a, inverse_mode=False), inverse_mode=True))

    # x = FlowStep(32)
    # print(x(a))

    # # check the batch normalization
    # bn = BatchNormalization()
    # a_ = bn(a, training=True)[0]
    # a__ = bn(a_, reverse=True)[0]
    # tf.assert_equal(a__, a)
    # exit()

    model = GLOW(SQUEEZE_FACTOR, K_GLOW, L_GLOW, IMG_SIZE, CHANNEL_SIZE)
    # import time
    # model(a)[0]
    # model.load_weights(CHECKPOINT_PATH+".h5")
    #
    # z = model(a)[0]
    # print(tf.reduce_min(z), tf.reduce_max(z))
    # a_1 = model(z, reverse=True)[0]
    # a_1 = my_tf_round(a_1, 3)  # round it
    #
    # print(tf.reduce_sum(tf.cast(a!=a_1, dtype=tf.float32)))
    #
    # tf.assert_equal(a_1, a)
    # print(model.sample())
    #
    # exit()

    # print(model(a, logdet=True, reverse=False))
    z = model(a, training=True)[0]
    z = model(a)[0]
    print(tf.reduce_min(z), tf.reduce_max(z))
    a_1 = model(z, reverse=True)[0]
    a_1 = my_tf_round(a_1, 3)  # round it

    print(tf.reduce_sum(tf.cast(a!=a_1, dtype=tf.float32)))

    tf.assert_equal(a_1, a)
    print(model.sample())

    # import csv
    # with open('mycsvfile.csv', 'w') as f:  # Just use 'w' mode in 3.x
    #     w = csv.DictWriter(f, model.debugging.keys())
    #     w.writeheader()
    #     w.writerow(model.debugging)
