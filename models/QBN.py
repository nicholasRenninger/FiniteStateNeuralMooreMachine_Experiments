import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    import sklearn.preprocessing as prep
    from tensorflow.examples.tutorials.mnist import input_data


# As a QBN is essentially an applied autoencoder, this base class was taken and
# modified from the tf 1 model examples at :
#
# https://github.com/tensorflow/models/tree/master/research/autoencoder

class QBN(object):

    def __init__(self, size_of_layers, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),
                 log_dir='logs'):

        self.size_of_layers = size_of_layers
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.size_of_layers[0]])

        # use x as the variable we encode and then decode with the
        # encoder / decoder modules
        h = self.x

        # build encoder module
        self.hidden_encode = []

        for layer in range(len(self.size_of_layers) - 1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)
        self.encoding = self.hidden_encode[-1]

        # build decoder module
        self.hidden_decode = []
        for layer in range(len(self.size_of_layers) - 1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
                       self.weights['recon'][layer]['b']))
            self.hidden_decode.append(h)
        self.decoding = self.hidden_decode[-1]

        # MSE Reconstruction Loss / loss
        self.reconstruction_diff = tf.subtract(self.decoding, self.x)
        self.loss = 0.5 * tf.reduce_sum(tf.pow(self.reconstruction_diff, 2.0))
        self.loss_summ = tf.summary.scalar('MSE Loss', self.loss)
        self.optimizer = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.size_of_layers) - 1):
            w = tf.Variable(initializer((self.size_of_layers[layer],
                                         self.size_of_layers[layer + 1]),
                                        dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.size_of_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})

        # decoder network weights
        recon_weights = []
        for layer in range(len(self.size_of_layers) - 1, 0, -1):
            w = tf.Variable(initializer((self.size_of_layers[layer],
                                         self.size_of_layers[layer - 1]),
                                        dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.size_of_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})

        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights

        return all_weights

    def partial_fit(self, X):
        """
        does a batch of gradient descent

        :param      X:    training data batch to fit

        :returns:   the batch training loss scalar
        """

        (loss, opt,
         loss_summary) = self.sess.run((self.loss, self.optimizer,
                                        self.loss_summ),
                                       feed_dict={self.x: X})

        return loss

    def fit(self, X, training_epochs=400, batch_size=32, n_samples=None,
            verbose=False, display_step=1, log_dir='logs'):
        """
        Fully fits the QBN to the training data X

        :param      X:                the training data
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
            writer = tf.summary.FileWriter(log_dir)

        # try to compute the number of samples if not given, but sometimes
        # len is not going to work.
        if n_samples is None:
            n_samples = len(X)

        for epoch in range(training_epochs):

            avg_loss = 0.
            total_batch = int(n_samples / batch_size)

            # Loop over all batches
            for i in range(total_batch):

                batch_xs = self.get_random_block_from_data(X, batch_size)

                # Fit training using batch data
                loss, _, loss_summary = autoencoder.partial_fit(batch_xs)

                # Compute average loss
                avg_loss += loss / n_samples * batch_size

                # update tensorboard logs with the loss from the most recent
                # batch of data
                if write_to_logs:
                    writer.add_summary(loss_summary, i)

            # Display logs per epoch step
            if (epoch % display_step == 0) and verbose:
                print("Epoch:", '%d,' % (epoch + 1),
                      "loss:", "{:.9f}".format(avg_loss))

    def get_random_block_from_data(self, data, batch_size):

        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]

    def calc_total_loss(self, X):
        return self.sess.run(self.loss, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden_encode[-1], feed_dict={self.x: X})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['encode'][-1]['b'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden_encode[-1]: hidden})

    def encode(self, X):
        return self.sess.run(self.encoding, feed_dict={self.x: X})

    def decode(self, X):
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
    training_epochs = 20
    batch_size = 128

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    autoencoder = QBN(size_of_layers=[784, 200],
                      transfer_function=tf.nn.softplus,
                      optimizer=optimizer)

    autoencoder.fit(X_train, training_epochs=training_epochs,
                    batch_size=batch_size, n_samples=n_samples, verbose=True)

    print("Total loss: " + str(autoencoder.calc_total_loss(X_test)))
