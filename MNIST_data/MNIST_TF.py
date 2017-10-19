import tensorflow as tf

logs_path = '/Volumes/Transcend/GitHub/TensorFlow_MNIST/'
# View Tensorboard: http://xinxuequs-mbp.student.iastate.edu:6006

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Setting Training Parameters
learning_rate = 0.001
training_epochs = 20000
batch_size = 100

# function to define fully connected layers: using input, input size, output size, and name_scope
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    #tf.summary.histogram("weights", w)
    #tf.summary.histogram("biases", b)
    #tf.summary.histogram("activations", act)
    return act

def mnist_model(learning_rate):
  tf.reset_default_graph()
  sess = tf.Session()

  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  # construct a NN with two hidden layers
  fc1 = fc_layer(x, 784, 200, "fc1")
  fc2 = fc_layer(fc1, 200, 50, "fc2")
  logits = fc_layer(fc2, 50, 10, "fc3")

  # define loss function using cross entropy
  with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="loss")
    # save the objective loss value into summary
    tf.summary.scalar("loss", loss)

  # train the model using AdamOptimizer
  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  # calculate accuracy and save it to summary
  with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(logs_path)
  writer.add_graph(sess.graph)

  # Batch training
  for i in range(training_epochs):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
        [TrainAccuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, i)
        print("Training Accuracy is:", TrainAccuracy)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

  # Test model
  [Taccuracy, s] = sess.run([accuracy, summ], feed_dict={x: mnist.test.images, y: mnist.test.labels})
  print("Testing Accuracy is:", Taccuracy)

mnist_model(learning_rate)