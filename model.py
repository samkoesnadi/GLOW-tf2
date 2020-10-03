from common_definitions import *
from utils.utils import *


class InvConv1(tf.keras.layers.Layer):
    """
    This is replacement of fixed permutation
    The weight has to be guaranteed to be square-sized, no bias
    """
    def __init__(self, channel_size):
        super().__init__()
        self.W = self.add_weight("W", shape=(channel_size, channel_size), initializer=tf.keras.initializers.Orthogonal(), regularizer=SOFT_KERNEL_REGULARIZER, trainable=True)

    def call(self, inputs, logdet=False, reverse=False):
        W = avoid_zero_function(self.W)

        if reverse:
            x = tf.einsum("ml,ijkl->ijkm", tf.linalg.inv(W), inputs)
        else:
            x = tf.einsum("ml,ijkl->ijkm", W, inputs)
        # print(tf.linalg.det(W))
        if logdet:
            return x, inputs.shape[1] * inputs.shape[2] * tf.math.log(tf.math.abs(tf.linalg.det(W)))
        else:
            return x, None

        # x = inputs[::-1]
        # if logdet:
        #     return x, 0
        # else:
        #     return x, None


class ActNormalization(tf.keras.layers.Layer):
    def __init__(self, channel_size):
        super().__init__()

        # !!! Do not change the order of the add weight
        self.s = self.add_weight("s", shape=channel_size, initializer=tf.keras.initializers.ones(), regularizer=SOFT_KERNEL_REGULARIZER, trainable=True)
        self.b = self.add_weight("b", shape=channel_size, initializer=tf.keras.initializers.zeros(), trainable=True)
        self.channel_size = channel_size

        # temp var
        self._initiated = False  # toggle var to initiate the value

    def call(self, inputs, logdet=False, reverse=False):
        if (not self._initiated) and (not reverse):
            std = tf.math.reduce_std(inputs, [0,1,2])
            mean = tf.math.reduce_mean(inputs, [0,1,2])
            self.s.assign(1/(std+TF_EPS))
            self.b.assign(-mean/(std+TF_EPS))

            self._initiated = True  # change the toggle var

        s = self.s[None, None, None, :]  # TODO: should be activated?
        b = self.b[None, None, None, :]  # range can be out of (-1, 1)

        # apply avoid zero function to s
        s = avoid_zero_function(s)

        if reverse:
            x = (inputs - b) / s
        else:
            x = s * inputs + b

        if logdet:
            return x, inputs.shape[1] * inputs.shape[2] * tf.reduce_sum(tf.math.log(tf.math.abs(s)))
        else:
            return x, None


class AffineCouplingLayer(tf.keras.layers.Layer):
    def __init__(self, channel_size):
        super().__init__()
        self.nn = self.nnLayer(channel_size)
        self.w = self.add_weight("w_external", shape=(channel_size,), trainable=True, initializer=KERNEL_INITIALIZER_CLOSE_ZERO, regularizer=HARD_KERNEL_REGULARIZER)
        self.b = self.add_weight("b_external", shape=(channel_size,), trainable=True, initializer=tf.keras.initializers.zeros())

        # variables
        self.channel_size = channel_size

    def nnLayer(self, channel_size):
        inputs = tf.keras.Input(shape=(None,None,channel_size//2))
        x = tf.keras.layers.Conv2D(channel_size // 4, 1, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=SOFT_KERNEL_REGULARIZER, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT_N)(x)
        x = tf.keras.layers.Conv2D(channel_size // 4, 3, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=SOFT_KERNEL_REGULARIZER, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT_N)(x)
        x = tf.keras.layers.Conv2D(channel_size // 2, 1, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=SOFT_KERNEL_REGULARIZER, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = x + inputs

        # x = tf.keras.layers.Conv2D(channel_size // 4, 3, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=SOFT_KERNEL_REGULARIZER, padding="same")(inputs)
        # x = tf.keras.layers.BatchNormalization()(x)
        # # x = tf.keras.layers.Dropout(DROPOUT_N)(x)
        # x = tf.keras.layers.Conv2D(channel_size // 4, 1, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=SOFT_KERNEL_REGULARIZER, padding="same")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # # x = tf.keras.layers.Dropout(DROPOUT_N)(x)

        # TODO: analyze these below
        s = tf.keras.layers.Conv2D(channel_size//2, 3, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER_CLOSE_ZERO,  kernel_regularizer=HARD_KERNEL_REGULARIZER, padding="same")(x)
        t = tf.keras.layers.Conv2D(channel_size//2, 3, activation=ACTIVATION, kernel_initializer=tf.keras.initializers.zeros(), padding="same")(x)

        s = avoid_zero_function(s)
        t = avoid_zero_function(t)

        return tf.keras.Model(inputs, [tf.exp(s), t])

    def forward_block(self, x, s, t, w, b):
        return w * (x * s + t) + b

    def backward_block(self, y, s, t, w, b):
        return (y - b - w * t) / (w * s)

    def call(self, inputs, logdet=False, reverse=False):
        xa, xb = split_last_channel(inputs)
        sb, tb = self.nn(xb)
        sa, ta = self.nn(xa)

        # preprocess w & b
        w = avoid_zero_function(self.w)
        w = tf.exp(w)
        wa, wb = w[:self.channel_size//2], w[self.channel_size//2:]

        ba, bb = self.b[:self.channel_size//2], self.b[self.channel_size//2:]

        # print(w)
        # print(sb)

        if reverse:
            ya = self.backward_block(xa, sb, tb, wa, ba)
            yb = self.backward_block(xb, sa, ta, wb, bb)
        else:
            ya = self.forward_block(xa, sb, tb, wa, ba)
            yb = self.forward_block(xb, sa, ta, wb, bb)

        if logdet:
            # _det = wa * wb * sa * sb
            _logabsdet = log_abs(wa) + log_abs(wb) + tf.reduce_mean(log_abs(sa), 0) + tf.reduce_mean(log_abs(sb), 0)
            return (ya, yb), tf.reduce_sum(_logabsdet)
        else:
            return (ya, yb), None


class FlowStep(tf.keras.layers.Layer):
    def __init__(self, channel_size):
        super().__init__()
        self.an = ActNormalization(channel_size)
        self.perm = InvConv1(channel_size)
        self.acl = AffineCouplingLayer(channel_size)

    def call(self, inputs, logdet=False, reverse=False):
        if not reverse:
            # act norm
            x, logdet_an = self.an(inputs, logdet, reverse)

            # invertible 1x1 layer
            x, logdet_perm = self.perm(x, logdet, reverse)

            # affine coupling layer
            x, logdet_acl = self.acl(x, logdet, reverse)
        else:
            # affine coupling layer
            x, _ = self.acl(inputs, logdet, reverse)
            x = tf.concat(x, axis=-1)  # concat the two output produced

            # invertible 1x1 layer
            x, _ = self.perm(x, logdet, reverse)

            # act norm
            x, _ = self.an(x, logdet, reverse)

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
                inputs, 0, 0, height//self.factor_size*self.factor_size, width//self.factor_size*self.factor_size
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
        sqrt_factor_size = int(factor_size**.5)  # sqrt the factor size as it is per dimension
        self.channel_order = [int(CHANNEL_SIZE*factor_size**(l+1)/2**l) for l in range(L)]  # channel order in the multi-scale architecture

        # layers
        # self.cropifnotfitlayer = CropIfNotFitLayer(sqrt_factor_size)
        self.squeezelayers = [SqueezeLayer(sqrt_factor_size) for _ in range(L)]
        self.flowsteps = [[FlowStep(c) for _ in range(K)] for c in self.channel_order]
        self.mean_logsd_nns = [tf.keras.layers.Dense(c, trainable=False, kernel_initializer=tf.keras.initializers.zeros(), kernel_regularizer=SOFT_KERNEL_REGULARIZER)
                               for c in self.channel_order]  # TODO: this might need to be configured

        # constant var
        self.factor_size = factor_size
        self.K = K
        self.L = L

        # self.debugging = {}

    # @tf.function
    def call(self, inputs, logdet=False, reverse=False):
        if not reverse:
            # # crop to matches the squeeze function
            # x = self.cropifnotfitlayer(inputs)
            x = inputs

            # run inner iteration of L-1 times
            z = []
            logdet_fs_total = 0

            for i_l in range(self.L):
                x = self.squeezelayers[i_l](x, reverse)
                # self.debugging[f"squeeze_{i_l}"] = [x[0,0,0,0].numpy()]

                # run flow step for K times
                for i_k in range(self.K - 1):
                    (ya, yb), logdet_fs = self.flowsteps[i_l][i_k](x, logdet, reverse)
                    x = concat_last_channel(yb, ya)  # flip the ya and yb as of the architecture design
                    # self.debugging[f"flow_{i_l}_{i_k}"] = [x[0,0,0,0].numpy()]
                    if logdet: logdet_fs_total += logdet_fs

                # run the last K without concat
                (ya, yb), logdet_fs = self.flowsteps[i_l][self.K-1](x, logdet, reverse)
                if logdet: logdet_fs_total += logdet_fs
                # self.debugging[f"flow_{i_l}_{self.K-1}"] = [ya[0,0,0,0].numpy()]

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
                # mean_logsd = self.mean_logsd_nns[i_l](yb)
                mean_var = tf.zeros(z_total.shape[1])
                mean = mean_var
                var = mean_var + 1
                # logpx = tf.expand_dims(flatten_sum(logpz(mean, logsd, ya)), -1) + tf.expand_dims(logdet_fs_total, 0)
                logpzs = logpz(mean, var, z_total)

                return z_total, logpzs / BATCH_SIZE + logdet_fs_total
            else:
                return z_total, None
        else:
            assert not logdet  # inv cant have logdet
            z_total = inputs
            z_shape = z_total.get_shape()
            z_sizes = [int(z_shape[1]/2**(i_l+1)) for i_l in range(self.L)]  # the sizes as effect to the multi-scale arch
            x = None

            for i_l, z_size in enumerate(z_sizes[::-1]):
                if i_l == 0:
                    z_size *= 2
                i_l = self.L - i_l - 1  # reverse the index

                z_total, z = split_last_channel(z_total, boundary=-z_size)  # get the z

                za_channel_size = self.channel_order[i_l] if i_l == self.L-1 else self.channel_order[i_l]//2
                wh_size = int((z_size / za_channel_size) ** .5)

                if i_l==self.L-1:
                    z1, z2 = split_last_channel(z)
                    z1 = tf.reshape(z1, [-1, wh_size, wh_size, za_channel_size//2])
                    z2 = tf.reshape(z2, [-1, wh_size, wh_size, za_channel_size//2])
                    z = concat_last_channel(z1, z2)
                else:
                    z = tf.reshape(z, [-1, wh_size, wh_size, za_channel_size])
                    z = concat_last_channel(z, x)  # concat the z and previous x

                # run the last K
                # self.debugging[f"flow_{i_l}_{self.K-1}"].append(z[0,0,0,0].numpy())
                x, _ = self.flowsteps[i_l][self.K-1](z, logdet, reverse)

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

            return x, None


if __name__ == "__main__":
    a = tf.random.uniform((256,32,32,1), 0, 1)
    # x = InvConv1(32)
    # print(x(x(a, inverse_mode=False), inverse_mode=True))

    # x = FlowStep(32)
    # print(x(a))

    model = GLOW(SQUEEZE_FACTOR,K_GLOW,L_GLOW)
    import time
    # print(model(a, logdet=True, reverse=False))
    model.load_weights("test.hdf5")
    a_1 = model(model(a)[0], reverse=True)[0]

    tf.assert_equal(a_1, a)


    # import csv
    # with open('mycsvfile.csv', 'w') as f:  # Just use 'w' mode in 3.x
    #     w = csv.DictWriter(f, model.debugging.keys())
    #     w.writeheader()
    #     w.writerow(model.debugging)