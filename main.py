from common_definitions import *
from pipeline import *
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Step 1. the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, CHANNEL_SIZE)
    x_test = x_test.reshape(x_test.shape[0], IMG_SIZE, IMG_SIZE, CHANNEL_SIZE)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(SHUFFLE_SIZE)\
        .map(augment_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Step 2. the brain
    brain = Brain(SQUEEZE_FACTOR, K_GLOW, L_GLOW, LEARNING_RATE)

    # Step 3. training iteration

    # define metrics variables
    nll = tf.keras.metrics.Mean("nll")
    test_img = x_train[0][None,:]
    test_img_2 = x_train[1][None,:]
    test_z = np.random.normal(0, 1, (1, IMG_SIZE * IMG_SIZE * CHANNEL_SIZE))
    plt.imsave("test_ref.png", np.squeeze(test_img))
    plt.imsave("test_ref2.png", np.squeeze(test_img_2))

    for ep in range(EPOCHS):
        print(f"epoch {ep+1}/{EPOCHS}")
        nll.reset_states()

        # iteration per epoch
        with tqdm(train_dataset) as t:
            for x_t, y_t in t:
                nll(brain.train_step(x_t))  # run the train step and store the nll in the variable
                t.set_postfix(nll=nll.result().numpy())

        # save weight every epoch
        brain.model.save_weights("test.hdf5")

        # store image for evaluation
        plt.imsave("test.png", np.minimum(np.maximum(
            np.squeeze(
                brain.model(
                    (brain.model(test_img)[0]+brain.model(test_img_2)[0])/2,
                    reverse=True)[0].numpy()
            ), 0), 1))
        plt.imsave("test2.png", np.minimum(np.maximum(
            np.squeeze(
                brain.model(
                    (brain.model(test_img)[0]+3*brain.model(test_img_2)[0])/4,
                    reverse=True)[0].numpy()
            ), 0), 1))
        plt.imsave("test3.png", np.minimum(np.maximum(
            np.squeeze(
                brain.model(
                    # brain.model(test_img, resample=True)[0],
                    test_z,
                    reverse=True)[0].numpy()
            ), 0), 1))
        plt.imsave("test4.png", np.minimum(np.maximum(
            np.squeeze(
                brain.model(
                    brain.model(test_img)[0]+test_z,
                    reverse=True)[0].numpy()
            ), 0), 1))
        plt.imsave("test5.png", np.minimum(np.maximum(
            np.squeeze(
                brain.model(
                    brain.model(test_img)[0],
                    reverse=True)[0].numpy()
            ), 0), 1))