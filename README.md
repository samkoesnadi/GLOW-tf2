# Glow: Generative Flow with Invertible 1x1 Convolutions in Tensorflow 2

![python 3](https://img.shields.io/badge/python-3-blue.svg)
![tensorflow 2](https://img.shields.io/badge/tensorflow-2-orange.svg)

![](assets/example_1.png)
![](assets/example_2.png)
![](assets/example_3.png)

My implementation of GLOW from the paper https://arxiv.org/pdf/1807.03039 in Tensorflow 2. GLOW is
an interesting generative model as it uses invertible neural network to transform images to normal distribution and vice versa.
Additionally, it is strongly based on RealNVP, so knowing it would be helpful to understand
GLOW's contribution.

##### Table of Contents  
- [Why?](#why)  
- [Requirements](#requirements)
- [Training](#training)
- [Sampling](#sampling)
- [Future improvements](#future-improvements)
- [CONTRIBUTING](#contributing)
- [LICENSE](#license)

---

## Why?
Normalizing flows is an interesting field of generative model as the optimization 
is derived from exact prior distribution of the images, as opposed to Variational Autoencoder (approximated using Evidence Lower Bound)
and Generative Adversarial Network (using Jensen-Shannon Divergence).

The author of the paper has implemented the original version in Tensorflow 1 (https://github.com/openai/glow).
However, with the current default version of Tensorflow 2, the repository is no longer actual. This brings
the need of Tensorflow 2 implementation. Furthermore, here is provided the bare minimum
of the algorithm which is easily modifiable. Simplicity is always the goal here and 
contribution is always welcome!

Note that the implementation is not exactly the same as what proposed in the paper mainly to
improve the algorithm. This small differences lie in the network architecture and training hyperparameters.

## Requirements
`pip3 install -r requirements.txt`

## Training
After every epoch, the network's weights will be stored in the checkpoints directory defined in `common_definitions.py`.
There are also some sampling of the network (image generation mode) that are going
to be stored in results directory. Additionally, TensorBoard is used to track z's mean and variance, as well as the negative log-likelihood.
In optimal state, z should have zero mean and one variance. Additionally, the TensorBoard stores sampling with temperature of 0.7.

```python3
python3 main.py [-h] [--dataset [DATASET]] [--k_glow [K_GLOW]] [--l_glow [L_GLOW]]
       [--img_size [IMG_SIZE]] [--channel_size [CHANNEL_SIZE]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset [DATASET]   The dataset to train on ("mnist", "cifar10", "cifar100")
  --k_glow [K_GLOW]     The amount of blocks per layer
  --l_glow [L_GLOW]     The amount of layers
  --img_size [IMG_SIZE] The width and height of the input images
  --channel_size [CHANNEL_SIZE]
                        The channel size of the input images
```

More parameters of the implementation can be found at `common_definitions.py`. The pretrained weight for Cifar-10 can be downloaded at https://github.com/samuelmat19/GLOW-tf2/releases/download/0.0.1/weights.h5 

## Sampling
Sample the network with temperature of default 1.0

```python3 
python3 sample.py [-h] [--temp [TEMP]]

optional arguments:
  -h, --help     show this help message and exit
  --temp [TEMP]  The temperature of the sampling
```

## Future improvements
- [ ] Clean project and set up proper CI (prioritized)
- [ ] Improve documentation
- [ ] Analyze instability of the network's training that occurs (Matrix Invertible when backpropagating to update weights)

## CONTRIBUTING
To contribute to the project, these steps can be followed. Anyone that contributes will surely be recognized and mentioned here!

Contributions to the project are made using the "Fork & Pull" model. The typical steps would be:

1. create an account on [github](https://github.com)
2. fork this repository
3. make a local clone
4. make changes on the local copy
5. commit changes `git commit -m "my message"`
6. `push` to your GitHub account: `git push origin`
7. create a Pull Request (PR) from your GitHub fork
(go to your fork's webpage and click on "Pull Request."
You can then add a message to describe your proposal.)


## LICENSE
This open-source project is licensed under MIT License.
