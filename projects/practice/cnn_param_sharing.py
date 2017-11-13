import tensorflow as tf

# Output depth
k_output = 64

# Image properties
image_height = 10
image_width = 10
color_channels = 3

# Convolution filter
filter_size_height = 5
filter_size_width = 5

# Input/image
input = tf.placeholder(tf.float32, shape=[None, image_height, image_width, color_channels])
print(input)

# Weight and bias
weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
print(weight)
bias = tf.Variable(tf.zeros(k_output))
print(bias)
strides = [1, 2, 2, 1]
padding = 'SAME'

conv_layer = tf.nn.conv2d(input, weight, strides, padding)
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
conv_layer = tf.nn.max_pool(conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

print(conv_layer)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(conv, feed_dict={inp: (2.0, _, _, _)}))

