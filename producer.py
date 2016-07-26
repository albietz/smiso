import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def random_index_input_producer(limit, num_epochs=None, seed=None,
                                capacity=32, shared_name=None, name=None):
    """Like tf.train.range_input_producer, but randomizes every index
    instead of taking a range and shuffling every epoch (random reordering)."""
    with ops.op_scope([limit], name, "input_producer") as name:
        index_tensor = tf.random_uniform([limit], minval=0, maxval=limit, dtype=tf.int64, seed=seed)
        return tf.train.input_producer(
            index_tensor, [], num_epochs, False, None, capacity,
            shared_name, name, "fraction_of_%d_full" % capacity)


def random_slice_input_producer(tensor_list, num_epochs=None, seed=None,
                                capacity=32, shared_name=None, name=None):
    """Like tf.train.slice_input_producer, but uses random_index_input_producer
    instead of range_input_producer."""
    with ops.op_scope(tensor_list, name, "input_producer"):
        tensor_list = ops.convert_n_to_tensor_or_indexed_slices(tensor_list)
        if not tensor_list:
              raise ValueError(
                  "Expected at least one tensor in slice_input_producer().")
        range_size = tf.to_int64(array_ops.shape(tensor_list[0])[0])
        queue = random_index_input_producer(
                    range_size, num_epochs=num_epochs,
                    seed=seed, capacity=capacity,
                    shared_name=shared_name)
        index = queue.dequeue()
        output = [array_ops.gather(t, index) for t in tensor_list]
        return output


if __name__ == '__main__':
    # q = tf.train.range_input_producer(3)
    q = random_index_input_producer(3, seed=7)
    index = q.dequeue()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        threads = tf.train.start_queue_runners(sess=sess)

        idxs = []
        for _ in range(100):
            idx = sess.run(index)
            idxs.append(idx)
            print(idx)

        from collections import Counter
        print(Counter(idxs))
