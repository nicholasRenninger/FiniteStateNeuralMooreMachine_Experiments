import tensorflow as tf
from tensorflow.python.framework import ops


def ternary_activation(x, slope_tensor=None, alpha=1):
    """
    Does not support slope annealing (slope_tensor is ignored)
    Wolfram Alpha plot:
    https://www.wolframalpha.com/input/?i=plot+(1.5*tanh(x)+%2B+0.5(tanh(-(3-1e-2)*x))),+x%3D+-2+to+2
    """
    return 1.5 * tf.tanh(alpha * x) + 0.5 * (tf.tanh(-(3 / alpha) * x))


#
# taken from https://r2rt.com/beyond-binary-ternary-and-one-hot-neurons.html
#
def n_ary_activation(x, activation=ternary_activation,
                     slope_tensor=None, stochastic_tensor=None):

    """
    n-ary activation for creating binary and ternary neurons (and n-ary
    neurons, if you can create the right activation function).

    Given a tensor and an activation, it:

    * applies the activation to the tensor
    * either samples the results (if stochastic tensor is true), or rounds the
    results (if stochastic_tensor is false) to the closest integer values.

    The default activation is a sigmoid (when slope_tensor = 1), which results
    in a binary neuron, as in:
    http://r2rt.com/binary-stochastic-neurons-in-tensorflow.html.

    Uses the straight through estimator during backprop. See:
    https://arxiv.org/abs/1308.3432.

    Arguments:
    * x: the pre-activation / logit tensor
    * activation: sigmoid, hard sigmoid, or n-ary activation
    * slope_tensor: slope adjusts the slope of the activation function,
                    for purposes of the Slope Annealing Trick
                    (see http://arxiv.org/abs/1609.01704)
    * stochastic_tensor: whether to sample the closest integer, or round to it.
    """

    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    if stochastic_tensor is None:
        stochastic_tensor = tf.constant(True)

    p = activation(x, slope_tensor)

    return tf.cond(stochastic_tensor,
                   lambda: sample_closest_ints(p),
                   lambda: st_round(p))


def st_round(x):
    """
    Rounds a tensor using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("StRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)


def sample_closest_ints(x):
    """
    If x is a float, then samples floor(x) with probability x - floor(x),
    and ceil(x) with probability ceil(x) - x, using the straight through
    estimator for the gradient.

    E.g.,:
    if x is 0.6, sample_closest_ints(x) will be 1 with probability 0.6, and
    0 otherwise, and the gradient will be pass-through (identity).
    """
    with ops.name_scope("SampleClosestInts") as name:
        grad_overide_map = {"Ceil": "Identity", "Sub": "SampleClosestInts"}
        with tf.get_default_graph().gradient_override_map(grad_overide_map):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)


@ops.RegisterGradient("SampleClosestInts")
def sample_closest_ints_grad(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]
