import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    import sklearn.preprocessing as prep
    from tensorflow.examples.tutorials.mnist import input_data
    from activations import n_ary_activation
else:
    from .activations import n_ary_activation


# As a QBN is essentially an applied autoencoder, this base class was taken and
# HEAVILY modified from the tf 1 model examples at:
#
# https://github.com/tensorflow/models/tree/master/research/autoencoder
# https://github.com/koulanurag/mmn/blob/65135da36dc8f8bfad05797f9d077c22adde0384/main_atari.py#L91

class QBN(object):
    """
    A quantized bottleneck network, which is really a quntizing
    autoencoder in disguise™

    This QBN reduces the dimensionality and quantizes its input features into a
    latent hidden state where each neuron has a discrete value, in this case
    from the ternary (three-values: -1, 0, 1) set.  Thus the combination of the
    latent states's values encode a ternary discrete value.

    :param      layer_sizes:                 the sizes of all network layers
                                             e.g. layer_sizes = [100, 80, 50]
                                             means the layers of the model are:
                                             * encoder_1 = (100, 80)
                                             * encoder_2 = (80, 50) <- latent
                                             * decoder_1 = (50, 80)
                                             * decoder_2 = (80, 100)
    :param      layers_activation_function:  All (encoder & decoder)
                                             hidden layers' activation
                                             function.
    :param      latent_activation_function:  The latent layer's activation
                                             function. This should be quantized
                                             so that the latent state is
                                             quantized.
    :param      optimizer:                   The optimizer
    :param      loss_func:                   The loss function for training
    :param      log_dir:                     The tensorboard logging directory
    """

    def __init__(self, layer_sizes,
                 layers_activation_function=tf.nn.tanh,
                 latent_activation_function=n_ary_activation,
                 optimizer=tf.train.AdamOptimizer(),
                 loss_func=tf.compat.v1.losses.mean_squared_error,
                 log_dir='logs'):

        self.layer_sizes = layer_sizes
        self.hidd_activation = layers_activation_function
        self.latent_activation = latent_activation_function
        self.input_dim = self.layer_sizes[0]

        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.x_test = tf.placeholder(tf.float32, [None, self.input_dim])

        # use x as the variable we encode and then decode with the
        # encoder / decoder modules
        h = self.x

        # build encoder module
        self.encode_layer_tensors = []
        latent_layer_idx = len(self.layer_sizes) - 2
        for layer in range(len(self.layer_sizes) - 1):

            # latent layer needs a quantized activation for this to be
            # quantized
            if layer == latent_layer_idx:
                activ = self.latent_activation
            else:
                activ = self.hidd_activation

            h = activ(tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                             self.weights['encode'][layer]['b']))
            self.encode_layer_tensors.append(h)

        self.encoding = self.encode_layer_tensors[-1]

        # build decoder module
        self.decode_layer_tensors = []
        for layer in range(len(self.layer_sizes) - 1):
            h = self.hidd_activation(
                tf.add(tf.matmul(h, self.weights['decode'][layer]['w']),
                       self.weights['decode'][layer]['b']))
            self.decode_layer_tensors.append(h)

        self.decoding = self.decode_layer_tensors[-1]

        # MSE Reconstruction Loss / loss
        self.train_loss = loss_func(self.x, self.decoding)
        self.test_loss = loss_func(self.x_test, self.decoding)

        # merge all summaries :)
        self.train_loss_summ = tf.summary.scalar('Train Loss', self.train_loss)
        self.test_loss_summ = tf.summary.scalar('Test Loss', self.test_loss)
        self.merged_summ = tf.compat.v1.summary.merge_all()

        self.optimizer = optimizer.minimize(self.train_loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.layer_sizes) - 1):
            w = tf.Variable(initializer((self.layer_sizes[layer],
                                         self.layer_sizes[layer + 1]),
                                        dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.layer_sizes[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})

        # decoder network weights
        recon_weights = []
        for layer in range(len(self.layer_sizes) - 1, 0, -1):
            w = tf.Variable(initializer((self.layer_sizes[layer],
                                         self.layer_sizes[layer - 1]),
                                        dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.layer_sizes[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})

        all_weights['encode'] = encoder_weights
        all_weights['decode'] = recon_weights

        return all_weights

    def partial_fit(self, train_data, test_data):
        """
        does a batch of gradient descent

        :param      train_data:  The training data
        :param      test_data:   The test data to evaluate the model on

        :returns:   losses and summaries for tb writing
                    (train_loss, test_loss, merged_summaries)
        """

        (train_loss, test_loss, opt,
         merged_summaries) = self.sess.run([self.train_loss,
                                            self.test_loss,
                                            self.optimizer,
                                            self.merged_summ],
                                           feed_dict={self.x: train_data,
                                                      self.x_test: test_data})

        return train_loss, test_loss, merged_summaries

    def fit(self, X, X_test,
            training_epochs=400, batch_size=32, n_samples=None,
            verbose=False, display_step=1, log_dir='logs'):
        """
        Fully fits the QBN to the training data X

        :param      X:                the training data
        :param      X_test:           the test data to evaluate the network on
        :param      training_epochs:  The training epochs
        :param      batch_size:       The training batch size
        :param      n_samples:        The number of samples in X
        :param      verbose:          controls whether or not to print live
                                      results
        :param      display_step:     How often to print loss updates if
                                      verbose
        :param      log_dir:          The tensorboard logging directory
                                      use None if you don't want to log
        """

        write_to_logs = (log_dir is not None)

        # for tensorboard logging of training
        if write_to_logs:
            writer = tf.compat.v1.summary.FileWriter(log_dir)

        # try to compute the number of samples if not given, but sometimes
        # len is not going to work.
        if n_samples is None:
            n_samples = len(X)

        for epoch in range(training_epochs):

            total_batch = int(n_samples / batch_size)

            # Loop over all batches
            for i in range(total_batch):

                batch_xs = self.get_random_block_from_data(X, batch_size)
                batch_xs_test = self.get_random_block_from_data(X_test,
                                                                batch_size)

                # Fit training using batch data
                (train_loss, test_loss,
                 merged_summaries) = self.partial_fit(train_data=batch_xs,
                                                      test_data=batch_xs_test)

            # update tensorboard logs with the loss from the most recent
            # batch of data
            if write_to_logs:
                writer.add_summary(merged_summaries, epoch)

            # Display logs per epoch step
            if (epoch % display_step == 0) and verbose:
                print("Epoch:", '%d,' % (epoch + 1),
                      "train_loss:", "{:.9f}".format(train_loss),
                      "test_loss:", "{:.9f}".format(test_loss))

    def get_random_block_from_data(self, data, batch_size):

        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def format_data(self, data):
        data_shape = data.shape
        if len(data_shape) > 1:
            return data
        else:
            reshaped_data = data.reshape((-1, self.input_dim))
            return reshaped_data

    def calc_total_loss(self, X):
        X = self.format_data(X)
        return self.sess.run(self.train_loss, feed_dict={self.x: X})

    def encode(self, X):
        X = self.format_data(X)
        return self.sess.run(self.encoding, feed_dict={self.x: X})

    def decode(self, X):
        X = self.format_data(X)
        return self.sess.run(self.decoding, feed_dict={self.x: X})

    def getWeights(self):
        raise NotImplementedError
        return self.sess.run(self.weights)

    def getBiases(self):
        raise NotImplementedError
        return self.sess.run(self.weights)


if __name__ == "__main__":

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        return X_train, X_test

    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)
    training_epochs = 10
    batch_size = 128

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
    quantized_autoencoder = QBN(layer_sizes=[784, 300, 200],
                                layers_activation_function=tf.nn.softplus,
                                optimizer=optimizer)

    quantized_autoencoder.fit(X_train, X_test, training_epochs=training_epochs,
                              batch_size=batch_size,
                              n_samples=n_samples, verbose=True)

    # example usage
    ex_data = X_train[0, :]
    latent_quantized_representation = quantized_autoencoder.encode(ex_data)
    reconstructed_data = quantized_autoencoder.decode(ex_data)
    print(ex_data[0:10])
    print(latent_quantized_representation.flatten()[0:10])
    print(reconstructed_data.flatten()[0:10])

    print("Total loss: " + str(quantized_autoencoder.calc_total_loss(X_test)))
