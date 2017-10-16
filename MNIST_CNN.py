import tensorflow as tf

logs_path = '/Volumes/Transcend/GitHub/TensorFlow_MNIST/'
# tensorboard --logdir=run1:/Volumes/Transcend/GitHub/TensorFlow_MNIST/ --port 6006
# View Tensorboard: http://xinxuequs-mbp.student.iastate.edu:6006

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Setting Training Parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = 100

def conv_layer(input, input_chs, output_chs, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([3, 3, input_chs, output_chs], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[output_chs]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="VALID") # w is the filter/kernel here
    act = tf.nn.relu(conv + b)
    return act
    # we do not need max-pooling here
    # tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    return act

def mnist_model(learning_rate):
  tf.reset_default_graph()
  sess = tf.Session()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  conv1 = conv_layer(x_image, 1, 64, "conv1")   # 1st C_layer
  conv2 = conv_layer(conv1, 64, 64, "conv2")    # 2nd C_layer
  conv_out = conv_layer(conv2, 64,64 , "conv3") # 3rd C_layer

  flattened = tf.reshape(conv_out, [-1, 22 * 22 * 64])

  fc1 = fc_layer(flattened, 22 * 22 * 64, 512, "fc1")

  # Output Layer
  logits = fc_layer(fc1, 512, 10, "fc2")

  with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="loss")
    tf.summary.scalar("loss", loss)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(logs_path)
  writer.add_graph(sess.graph)


  for i in range(training_epochs):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

  # Test model
  [Taccuracy, s] = sess.run([accuracy, summ], feed_dict={x: mnist.test.images, y: mnist.test.labels})
  print("Testing Accuracy is:", Taccuracy)

mnist_model(learning_rate)