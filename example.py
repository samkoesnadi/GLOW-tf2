import tensorflow as tf
import tensorflow_probability as tfp

x = tf.random.normal((512,512), dtype=tf.float32)
m = tf.random.normal((512,512), dtype=tf.float32)

mult = 1

x *= mult
m *= 1

y = m * x
x_o = (y) / (m)

# tf.assert_equal(x, x_o)

print(x, x_o)

print(1/512 * (1/512))
print(1/(512*512))