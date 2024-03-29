import numpy as np
import tensorflow as tf


CKN_MODEL = '/scratch/clear/abietti/results/ckn/mnist_py/layers_1.npy'


def load_dataset(num_transformations=0):
    from infimnist import _infimnist as imnist
    mnist = imnist.InfimnistGenerator()
    digits_train, labels_train = mnist.gen(np.arange(10000, 70000 + num_transformations * 60000))  # training digits
    digits_test, labels_test = mnist.gen(np.arange(10000))  # test digits
    def getX(digits):
        return digits.reshape(-1, 28, 28, 1).astype(np.float32)
    return (getX(digits_train), labels_train.astype(np.int32),
            getX(digits_test), labels_test.astype(np.int32))


def load_ckn_layers():
    return np.load(CKN_MODEL)


def params():
    return {
        'n_classes': 10,
        'lmbda': [5e-8],
        'lrs': [0.1, 0.2, 1.0],
        # 'miso_lrs': [1.0],
        'saga_lrs': [0.2],
        'results_root': '/scratch/clear/abietti/results/ckn/mnist_py/accs',
        'encode_size': 60000,
    }


def augmentation(image, sz=28):
    image = tf.image.resize_image_with_crop_or_pad(image, sz+8, sz+8)
    offset = tf.random_uniform([1], maxval=16, dtype=tf.int32)[0]
    image = tf.random_crop(image, size=[sz-offset, sz-offset, 1])
    image = tf.image.resize_images(image, sz, sz)
    image.set_shape([sz, sz, 1])
    return image


def augmentation_pad(image):
    # pad with zeros to make the image 32x32
    image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)

    # extract random 28x28 crops
    image = tf.random_crop(image, size=[28, 28, 1])

    return image
