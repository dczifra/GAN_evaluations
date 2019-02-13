import tensorflow as tf
hungarian_module = tf.load_op_library('./munkres-tensorflow/hungarian.so')
with tf.Session(''):
  print(hungarian_module.hungarian([[[1, 2], [3, 4]]]).eval())
