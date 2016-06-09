# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
DIM_FEATURES = 512  # size of the last layer (feature map)
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 100
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean("train_cnn", True, "True for training the CNN feature representation.")
tf.app.flags.DEFINE_boolean("train_sgd", False, "SGD")
tf.app.flags.DEFINE_boolean("train_svrg", False, "SVRG")
tf.app.flags.DEFINE_boolean("train_miso", False, "MISO")
FLAGS = tf.app.flags.FLAGS


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def tf_error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(predictions, 1), labels))))


class CNNModel(object):
  def __init__(self):
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    self.conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED), name='conv1_w')
    self.conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_b')
    self.conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED), name='conv2_w')
    self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name='conv2_b')
    self.fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, DIM_FEATURES],
            stddev=0.1,
            seed=SEED), name='fc1_w')
    self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[DIM_FEATURES]), name='fc1_b')
    self.fc2_weights = tf.Variable(
        tf.truncated_normal([DIM_FEATURES, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED), name='fc2_w')
    self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]), name='fc2_b')

    self.variables = [self.conv1_weights, self.conv1_biases, self.conv2_weights, self.conv2_biases,
                      self.fc1_weights, self.fc1_biases, self.fc2_weights, self.fc2_biases]

  def forward(self, data, dropout=False, last_layer=False):
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        self.conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        self.conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if dropout:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

    if last_layer:  # return feature vector only
      return hidden

    return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases


class Dataset(object):
  def __init__(self, augmentation=True, num_epochs=None, num_threads=10, capacity=256):
    self.num_epochs = num_epochs
    if FLAGS.self_test:
      print('Running self-test.')
      self.train_data, self.train_labels = fake_data(256)
      self.validation_data, self.validation_labels = fake_data(EVAL_BATCH_SIZE)
      self.test_data, self.test_labels = fake_data(EVAL_BATCH_SIZE)
    else:
      # Get the data.
      train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
      train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
      test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
      test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

      # Extract it into numpy arrays.
      train_data = extract_data(train_data_filename, 60000)
      train_labels = extract_labels(train_labels_filename, 60000)
      self.test_data = extract_data(test_data_filename, 10000)
      self.test_labels = extract_labels(test_labels_filename, 10000)

      # Generate a validation set.
      self.validation_data = train_data[:VALIDATION_SIZE, ...]
      self.validation_labels = train_labels[:VALIDATION_SIZE]
      self.train_data = train_data[VALIDATION_SIZE:, ...]
      self.train_labels = train_labels[VALIDATION_SIZE:]

    self.validation_features = None
    self.test_features = None

    # create the queue
    self.images_initializer = tf.placeholder(dtype=self.train_data.dtype,
                                             shape=self.train_data.shape)
    self.labels_initializer = tf.placeholder(dtype=self.train_labels.dtype,
                                             shape=self.train_labels.shape)
    self.indexes_initializer = tf.placeholder(dtype=tf.int64,
                                              shape=[self.train_data.shape[0]])
    self.input_images = tf.Variable(self.images_initializer, trainable=False, collections=[])
    self.input_labels = tf.Variable(self.labels_initializer, trainable=False, collections=[])
    self.input_indexes = tf.Variable(self.indexes_initializer, trainable=False, collections=[])

    image, label, index = tf.train.slice_input_producer(
        [self.input_images, self.input_labels, self.input_indexes],
        num_epochs=self.num_epochs, shuffle=True, seed=SEED)
    if augmentation:
      image = self.process_train_image(image)

    self.images, self.labels, self.indexes = tf.train.batch(
        [image, label, index], batch_size=BATCH_SIZE, num_threads=num_threads, capacity=capacity)


  def init(self, sess):
    sess.run(self.input_images.initializer,
             feed_dict={self.images_initializer: self.train_data})
    sess.run(self.input_labels.initializer,
             feed_dict={self.labels_initializer: self.train_labels})
    sess.run(self.input_indexes.initializer,
             feed_dict={self.indexes_initializer: range(self.train_labels.shape[0])})

  def init_test_features(self, eval_data, eval_op, sess):
    '''Initialize feature vectors on validation/test data from a given model.'''

    def eval_in_batches(data):
      """Get all last layer features for a dataset by running it in small batches."""
      size = data.shape[0]
      if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
      features = numpy.ndarray(shape=(size, DIM_FEATURES), dtype=numpy.float32)
      for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
          features[begin:end, :] = sess.run(
              eval_op,
              feed_dict={eval_data: data[begin:end, ...]})
        else:
          batch_features = sess.run(
              eval_op,
              feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
          features[begin:, :] = batch_features[begin - size:, :]
      return features

    self.validation_features = eval_in_batches(self.validation_data)
    self.test_features = eval_in_batches(self.test_data)

  def process_train_image(self, image):
    # pad with zeros to make the image 32x32
    image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)

    # extract random 28x28 crops
    image = tf.random_crop(image, size=[28, 28, 1])

    return image

def train_cnn():
  num_epochs = 1 if FLAGS.self_test else NUM_EPOCHS
  ds = Dataset(num_epochs=num_epochs)
  train_size = ds.train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.

  # train_data_node = tf.placeholder(
  #     tf.float32,
  #     shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  # train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  train_data_node = ds.images
  train_labels_node = ds.labels
  train_indexes_node = ds.indexes
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  m = CNNModel()

  # Training computation: logits + cross-entropy loss.
  logits = m.forward(train_data_node, dropout=True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(m.fc1_weights) + tf.nn.l2_loss(m.fc1_biases) +
                  tf.nn.l2_loss(m.fc2_weights) + tf.nn.l2_loss(m.fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  saver = tf.train.Saver(m.variables)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)
  train_error_rate = tf_error_rate(train_prediction, train_labels_node)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(m.forward(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    ds.init(sess)
    print('Initialized!')
    threads = tf.train.start_queue_runners(sess=sess)

    # Loop through training steps.
    for step in xrange(int(ds.num_epochs * train_size) // BATCH_SIZE):
      # Run the graph and fetch some of the nodes.
      _, l, lr, train_er = sess.run(
          [optimizer, loss, learning_rate, train_error_rate])
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % train_er)
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(ds.validation_data, sess), ds.validation_labels))
        sys.stdout.flush()

    # Finally print the result!
    test_error = error_rate(eval_in_batches(ds.test_data, sess), ds.test_labels)
    print('Test error: %.1f%%' % test_error)
    saver.save(sess, 'models/mnist')
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)


class EpochRunner(object):
  dataset_cls = Dataset
  model_cls = CNNModel
  model_filename = 'models/mnist'

  def __init__(self, compute_test_features=True):
    self.ds = self.dataset_cls()
    self.train_size = self.ds.train_labels.shape[0]

    self.m = self.model_cls()

    self.feats = self.m.forward(self.ds.images, last_layer=True)
    self.saver = tf.train.Saver(self.m.variables)

    # for computing test features
    if compute_test_features:
      eval_data = tf.placeholder(
          tf.float32,
          shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
      eval_features = self.m.forward(eval_data, last_layer=True)

    self.sess = tf.Session()
    self.saver.restore(self.sess, self.model_filename)
    self.ds.init(self.sess)
    tf.train.start_queue_runners(sess=self.sess)

    # compute test features
    if compute_test_features:
      self.ds.init_test_features(eval_data, eval_features, self.sess)

  def __enter__(self):
    self.sess.__enter__()

    return self

  def iter_epoch(self):
    assert self.sess is not None

    start_time = time.time()
    # Loop through training steps.
    # TODO: deal with incomplete batch in the end
    for step in xrange(self.train_size // BATCH_SIZE):
      features, labels, indexes = self.sess.run(
          [self.feats, self.ds.labels, self.ds.indexes])
      yield (features, labels, indexes)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / self.train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))

  def __exit__(self, ty, val, traceback):
    self.sess.__exit__(ty, val, traceback)


def run_on_train_data(func, num_epochs=1):
  ds = Dataset(num_epochs=num_epochs)
  train_size = ds.train_labels.shape[0]

  m = CNNModel()

  feats = m.forward(ds.images, last_layer=True)
  saver = tf.train.Saver(m.variables)
  start_time = time.time()
  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver.restore(sess, 'models/mnist')
    ds.init(sess)
    tf.train.start_queue_runners(sess=sess)

    # Loop through training steps.
    for step in xrange(int(ds.num_epochs * train_size) // BATCH_SIZE):
      features, labels, indexes = sess.run(
          [feats, ds.labels, ds.indexes])
      func(features, labels, indexes, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))


def train_svrg():
  pass


def train_miso():
  pass


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.train_cnn:
    train_cnn()
  elif FLAGS.train_svrg:
    train_svrg()
  elif FLAGS.train_miso:
    train_miso()


if __name__ == '__main__':
  tf.app.run()
