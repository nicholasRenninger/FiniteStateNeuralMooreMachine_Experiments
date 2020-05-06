import tensorflow as tf
from stable_baselines.common.base_class import TensorboardWriter
from stable_baselines.common.tf_util import make_session
from contextlib import suppress
import math

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
    A quantized bottleneck network, which is really a quantizing
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
    :param      decode_activation_function   activation function on the final
                                             layer of the QBN which is supposed
                                             to be the reconstructed input
    :param      optimizer:                   The training optimizer
    :param      gradient_clip_val:           global norm to clip all gradients
    :param      loss_func:                   The loss function for training
    """

    def __init__(self, layer_sizes,
                 layers_activation_function=tf.nn.tanh,
                 latent_activation_function=n_ary_activation,
                 decode_activation_function=tf.nn.relu6,
                 optimizer=tf.train.AdamOptimizer(),
                 gradient_clip_val=5.0,
                 loss_func=tf.compat.v1.losses.mean_squared_error,
                 net_name='qbn'):

        self.net_name = net_name

        self.hidd_activation = layers_activation_function
        self.latent_activation = latent_activation_function
        self.reconstruction_activation = decode_activation_function

        self.loss_func = loss_func
        self.optimizer = optimizer

        self.layer_sizes = layer_sizes
        self.input_dim = self.layer_sizes[0]

        self.grad_clip_val = gradient_clip_val

        self._setup_model()

    def _setup_model(self):

        self.graph = tf.Graph()
        self.sess = make_session(graph=self.graph)

        with self.graph.as_default():

            network_weights = self._initialize_weights()
            self.weights = network_weights

            self.x = tf.placeholder(tf.float32, [None, self.input_dim])

            # use x as the variable we encode and then decode with the
            # encoder / decoder modules
            h = self.x

            # use encoder module
            self.encoding = self._encode_ops(h)

            # use the decoder module
            self.decoding = self._decode_ops(self.encoding)

            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):

                # MSE Reconstruction Loss / loss
                self.train_loss = self.loss_func(self.x, self.decoding)
                self.train_loss_summ = tf.summary.scalar('Train_Loss',
                                                         self.train_loss)

                # need to clip gradients for training stability
                grad_info = self.optimizer.compute_gradients(self.train_loss)
                self.gradients, self.variables = zip(*grad_info)
                self.gradients, _ = tf.clip_by_global_norm(self.gradients,
                                                           self.grad_clip_val)
                clipped_grads = zip(self.gradients, self.variables)
                self.optimize = self.optimizer.apply_gradients(clipped_grads)

            tf.global_variables_initializer().run(session=self.sess)

            # for SAVING the model :)
            self.saver = tf.train.Saver()

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(self.net_name, reuse=tf.AUTO_REUSE):

            # Encoding network weights
            with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):

                encoder_weights = []
                for layer in range(len(self.layer_sizes) - 1):

                    # layer weight tensor
                    init_w = initializer((self.layer_sizes[layer],
                                          self.layer_sizes[layer + 1]),
                                         dtype=tf.float32)
                    w = tf.get_variable(f'w{layer}', initializer=init_w)

                    # layer bias tensor
                    init_b = tf.zeros([self.layer_sizes[layer + 1]],
                                      dtype=tf.float32)
                    b = tf.get_variable(f'b{layer}', initializer=init_b)

                    encoder_weights.append({'w': w, 'b': b})

            # decoder network weights
            with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):

                decoder_weights = []
                for layer in range(len(self.layer_sizes) - 1, 0, -1):

                    # layer weight tensor
                    init_w = initializer((self.layer_sizes[layer],
                                          self.layer_sizes[layer - 1]),
                                         dtype=tf.float32)
                    w = tf.get_variable(f'w{layer}', initializer=init_w)

                    # layer bias tensor
                    init_b = tf.zeros([self.layer_sizes[layer - 1]],
                                      dtype=tf.float32)
                    b = tf.get_variable(f'b{layer}', initializer=init_b)

                    decoder_weights.append({'w': w, 'b': b})

            all_weights['encode'] = encoder_weights
            all_weights['decode'] = decoder_weights

        return all_weights

    def _encode_ops(self, x):
        """
        runs ops for encoder module.

        returns the encoded latent state for the network input x
        """

        h = x

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

        return self.encode_layer_tensors[-1]

    def _decode_ops(self, h):
        """
        runs ops for decoder module.

        returns the decoded version of the latent state h
        """

        self.decode_layer_tensors = []
        reconstruction_layer_idx = len(self.layer_sizes) - 2
        for layer in range(len(self.layer_sizes) - 1):

            # reconstruction (final) layer needs a relu6 as per the paper
            if layer == reconstruction_layer_idx:
                activ = self.reconstruction_activation
            else:
                activ = self.hidd_activation

            h = activ(tf.add(tf.matmul(h, self.weights['decode'][layer]['w']),
                             self.weights['decode'][layer]['b']))
            self.decode_layer_tensors.append(h)

        return self.decode_layer_tensors[-1]

    def partial_fit(self, train_data, test_data):
        """
        does a batch of gradient descent

        :param      train_data:    The training data
        :param      test_data:     The test data to evaluate the model on

        :returns:   losses and tb summaries
                    (train_loss, test_loss, train_loss_summ, test_loss_summ)
        """

        (train_loss, opt,
         train_loss_summ) = self.sess.run([self.train_loss,
                                           self.optimize,
                                           self.train_loss_summ],
                                          feed_dict={self.x: train_data})

        (test_loss,
         test_loss_summ) = self.sess.run([self.train_loss,
                                          self.train_loss_summ],
                                         feed_dict={self.x: test_data})

        return train_loss, test_loss, train_loss_summ, test_loss_summ

    def fit(self, X, X_test,
            training_epochs=400, batch_size=32, n_samples=None,
            verbose=False, display_step=1, log_dir='logs',
            should_save=False, save_path='model.ckpt'):
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
        if write_to_logs:
            tbw_train = TensorboardWriter(self.graph, log_dir, 'train')
            tbw_test = TensorboardWriter(self.graph, log_dir, 'test')
        else:
            tbw_train = suppress()
            tbw_test = suppress()

        # try to compute the number of samples if not given, but sometimes
        # len is not going to work.
        if n_samples is None:
            n_samples = len(X)

        with tbw_test as test_writer, tbw_train as train_writer:
            for epoch in range(training_epochs):

                num_batches = math.ceil(n_samples / batch_size)

                # Loop over all batches
                for i in range(num_batches):

                    batch_start_idx = (i * batch_size)
                    batch_end_idx = (i * batch_size) + batch_size
                    batch_xs = X[batch_start_idx:batch_end_idx]
                    batch_xs_tst = X_test[batch_start_idx:batch_end_idx]

                    # Fit training using batch data
                    (train_loss, test_loss,
                     train_loss_summ,
                     test_loss_summ) = self.partial_fit(train_data=batch_xs,
                                                        test_data=batch_xs_tst)

                # update tensorboard logs with the loss from the most recent
                # batch of data
                if write_to_logs:
                    train_writer.add_summary(train_loss_summ, epoch)
                    test_writer.add_summary(test_loss_summ, epoch)

                # Display logs per epoch step
                if (epoch % display_step == 0) and verbose:
                    print("Epoch:", '%d,' % (epoch + 1),
                          "train_loss:", "{:.9f}".format(train_loss),
                          "test_loss:", "{:.9f}".format(test_loss))

        with self.sess as sess:
            self.saver.save(sess, save_path)

    def load_model(self, save_path):

        tf.reset_default_graph()

        self._setup_model()

        with self.sess as sess:
            self.saver.restore(sess, save_path)

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
