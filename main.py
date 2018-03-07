import tensorflow as tf

'''
Feed Forward

1. input > weights
> hidden layer 1 (activation function) > weights
> hidden layer 2 (activation function) > weights
> output layer

2. compare output to intended output with cost function (cross entropy)

Back Propogation

3. use optimizer > minimize cost (Adam Optimizar... SGD, AdaGrad...)
this goes backwards and manipulates weights, is called back propogation

---

feed forward + back propogation = 1 epoch

Keep cycling epochs for improvement

'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

'''

One Hot
Normally have 4 classes [0-3]

Onehot makes the data look like

0 = [1,0,0,0]
1 = [0,1,0,0]
2 = [0,0,1,0]
3 = [0,0,0,1]

'''

# neurons per layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# number of different types of images
n_classes = 10

# Feed 100 features at 1 time through the network
batch_size = 100

# input data
# images are 28x28 and we are flatening them so the length is 784
# So None (0) for height
# [x,y] dimensions for a matrix
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Placeholders are just a place for data at any given time to be put through the network

def netural_network_model(data):
  # The formula for the DNN is
  # (data * weights) + bias

  # 1. Set up all the Nodes (Neurons) and connect the layers

  # Setup the weights, number of pixels * number of neurons
  hidden_1_layer = {
    # First set of weights are random
    'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    # Biases just stop the return values from data * weights from being 0, because you cant multipy 0 by anything
    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
  }

  hidden_2_layer = {
    # Layer 2's input is the output of layer 1
    'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    # Biases just stop the return values from data * weights from being 0, because you cant multipy 0 by anything
    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
  }

  hidden_3_layer = {
    # Layer 3's input is the output of layer 2
    'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    # Biases just stop the return values from data * weights from being 0, because you cant multipy 0 by anything
    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
  }

  output_layer = {
    # Output layer is the number of nodes in layer 3 and the number classes
    'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    # Biases just stop the return values from data * weights from being 0, because you cant multipy 0 by anything
    'biases': tf.Variable(tf.random_normal([n_classes]))
  }

  # 1. Use the formula, stringing to together the output of each layer into the next layer
  # (data * weights) + bias

  l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])

  # Use the activation function on layer 1
  l1 = tf.nn.relu(l1)

  # Feed layer 1 into layer 2
  l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])

  # Use the activation function on layer 2
  l2 = tf.nn.relu(l2)

  # Feed layer 2 into layer 3
  l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])

  # Use the activation function on layer 3
  l3 = tf.nn.relu(l3)

  # Feed layer 3 into the output
  output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

  # Return the result
  return output

def train_neural_network(x):
  # Get a prediction for this data from the network
  # x is the data
  prediction = netural_network_model(x)

  # Use cross entropy with logits as our cost function
  # calculate the difference of the prection that we got with the known label that we have
  # y is the known label
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

  # Now we want to minimize the cost, which makes the prediction better
  '''
    If a child sees 10 examples of cats and all of them have orange fur,
    it will think that cats have orange fur and will look for orange fur
    when trying to identify a cat. Now it sees a black a cat and her parents
    tell her it's a cat (supervised learning). With a large “learning rate”,
    it will quickly realize that “orange fur” is not the most important feature
    of cats. With a small learning rate, it will think that this black cat is
    an outlier and cats are still orange.
  '''
  optimizer = tf.train.AdamOptimizer().minimize(cost)

  # How many epoch iterations we want to run, (cycles of feed forward + back propogation)
  hm_epochs = 500

  # Run the session
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Run through all the pochs
    for epoch in range(hm_epochs):
      epoch_loss = 0
      # How many batches do we need to run to get through all the examples
      for _ in range(int(mnist.train.num_examples / batch_size)):
        # Get the batch
        # x = data
        # y = labels
        epoch_x, epoch_y = mnist.train.next_batch(batch_size)

        print(epoch_x[0])
        # c = cost for this run
        # feed dict = just the structure of the data the session needs
        _, c = sess.run([optimizer, cost], feed_dict = { x: epoch_x, y: epoch_y })
        # Sum all the losses together for this epoch
        epoch_loss += c

      # Return the index of the item with the max value, because they are one hot [0,1,0] we can do this
      # They should be identical
      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

      # Grab the accuracy value of the returned correct item
      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

      # Compare the accuracy with all the test images and labels
      calculated_accuracy = accuracy.eval({ x: mnist.test.images, y: mnist.test.labels })

      # print some stats
      print('Epoch:', epoch + 1, '/', hm_epochs, 'Accuracy:',  calculated_accuracy, 'Loss:', epoch_loss)

train_neural_network(x)
