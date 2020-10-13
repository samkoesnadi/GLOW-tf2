from common_definitions import *
from pipeline import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Glow: Generative Flow with Invertible 1x1 Convolutions",
                                     description="My implementation of GLOW from the paper https://arxiv.org/pdf/1807.03039 in Tensorflow 2")
    parser.add_argument('--temp', type=float, nargs='?', default=1.0,
                        help='The temperature of the sampling')

    parser.print_help()  # print the help

    args = parser.parse_args()


    # Step 1. the brain
    brain = Brain(SQUEEZE_FACTOR, K_GLOW, L_GLOW, IMG_SIZE, CHANNEL_SIZE, LEARNING_RATE)

    # load weight if available
    brain.model(tf.random.uniform((2, IMG_SIZE, IMG_SIZE, CHANNEL_SIZE), 0.05, 1), training=True)
    print(brain.load_weights(CHECKPOINT_PATH))

    # Step 2. sample the network
    while True:
        plt.imshow(tf.clip_by_value(tf.squeeze(brain.sample(args.temp)), 0, 1).numpy())
        plt.show()