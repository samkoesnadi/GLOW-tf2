from common_definitions import *
import datetime


def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height // factor, factor,
                       width // factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height // factor, width //
                       factor, n_channels * factor * factor])
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels / factor ** 2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height * factor),
                       int(width * factor), int(n_channels / factor ** 2)))
    return x


def concat_last_channel(x, y):
    return tf.concat([x, y], -1)


def split_last_channel(x, boundary=None):
    x_shape = x.get_shape()
    boundary = math.ceil(x_shape[-1] / 2) if boundary is None else int(boundary)

    if len(x_shape) == 2:
        return x[:, :boundary], x[:, boundary:]
    else:
        return x[:, :, :, :boundary], x[:, :, :, boundary:]


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 4:
        return tf.reduce_sum(logps, [1, 2, 3])
    else:
        raise Exception()

def avoid_zero_function(x, a=90, b=0.05):
    return b * tf.nn.tanh(a*x) + x

def div_s(x, a, max_val=1e9):
    """div to scalar"""
    return tf.minimum(tf.maximum(x/a, -max_val), max_val)

def inv_elu(y):
    return tf.where(y < 0, tf.math.log(y+1+TF_EPS), y)

def d_elu(x):
    """
    derivative of elu
    :param x:
    :return:
    """
    return tf.where(x < 0, tf.exp(x), 1)

def log_abs(x):
    return tf.math.log(tf.math.abs(x))

def logpz(mean, var, x):
    """

    :param mean: (n)
    :param var: (n)
    :param x: (bsxn)
    :return: ()
    """
    # return -0.5 * tf.math.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd)
    # return -0.5 * (tf.math.log(2 * np.pi) - logsd) - .5 * ((x - mean) / tf.exp(logsd)) ** 2
    var_matrix = tf.linalg.diag(var)
    k = mean.shape[0]
    mean = mean[None, :]  # because there is no dimensionalty in the mean
    # tf.print(-.5*(x-mean)@var_matrix@tf.transpose(x-mean))
    logpz = -.5*tf.linalg.diag_part((x-mean)@var_matrix@tf.transpose(x-mean))\
            -(.5*(k*tf.math.log(2*np.pi)+tf.linalg.logdet(var_matrix)))
    # print(logpz)
    return tf.reduce_sum(logpz)

def pz(mean, var, x):
    var_matrix = tf.linalg.diag(var)
    k = mean.shape[0]
    mean = mean[None, :]  # because there is no dimensionalty in the mean
    # tf.print(-.5*(x-mean)@var_matrix@tf.transpose(x-mean))
    logpz = tf.exp(-.5*tf.linalg.diag_part((x-mean)@var_matrix@tf.transpose(x-mean))) / (tf.math.sqrt((2*np.pi)**k*tf.linalg.det(var_matrix)) + TF_EPS)
    # print(logpz)
    return tf.reduce_prod(logpz)

def relu1(x):
    return tf.math.minimum(x, 1)

def inv_sigmoid(x):
    return - tf.math.log(1/x-1)

def dev_sigmoid(x):
    """derivative of sigmoid"""
    return tf.nn.sigmoid(x)*(1-tf.nn.sigmoid(x))

def augment_data(image, label):
  # Add 6 pixels of padding
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
   # Random crop back to the original size
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, CHANNEL_SIZE])
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.clip_by_value(image, 0, 1)
  return image, label

class Tensorboard:
    def __init__(self, log_dir):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = log_dir + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def __call__(self, epoch, reward, actions_squared, Q_loss, A_loss):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('reward', reward.result(), step=epoch)
            tf.summary.scalar('actions squared', actions_squared.result(), step=epoch)
            tf.summary.scalar('critic loss', Q_loss.result(), step=epoch)
            tf.summary.scalar('actor loss', A_loss.result(), step=epoch)


if __name__ == "__main__":
    mean = np.zeros((10,), dtype=np.float32)
    var = np.ones((10,), dtype=np.float32)
    x = np.random.normal(0, 1, (400,10)).astype(np.float32)
    print(tf.exp(logpz(mean, var, x)))
    print(pz(mean, var, x))
    print(logpz(mean, var, x))