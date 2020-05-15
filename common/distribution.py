import tensorflow.compat.v1 as tf1
import numpy as np


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)

    def get_shape(self):
        return self.flatparam().shape

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector, init_scale, init_bias):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf1.placeholder(dtype=tf1.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf1.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.__dict__ == other.__dict__)


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf1.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf1.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf1.reduce_sum(tf1.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf1.to_float(tf1.shape(x)[-1]) \
               + tf1.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf1.reduce_sum(other.logstd - self.logstd + (tf1.square(self.std) + tf1.square(self.mean - other.mean)) / (
                    2.0 * tf1.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf1.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf1.random_normal(tf1.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf1.float32