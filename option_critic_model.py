import tensorflow.compat.v1 as tf1
import common.tf_util as U
import numpy as np
from common.distribution import DiagGaussianPdType
import gym
from common.mpi_running_mean_std import RunningMeanStd
import math
from model_learning import partialHybridModel
import random


def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf1.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf1.matmul(x, w[option[0]])
    if bias:
        b = tf1.get_variable(name + "/b", [num_options, size], initializer=tf1.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf1.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf1.get_variable_scope().name

    def _init(self, ob_space, ac_space, model, hid_size, num_hid_layers, num_options=2, term_prob=0.5, eps=0.0005):
        assert isinstance(ob_space, gym.spaces.Box)
        self.state_in = []
        self.state_out = []
        self.term_prob = term_prob
        self.num_options = num_options
        # Creating the policy network
        sequence_length = None
        self.ac_dim = ac_space.shape[0]
        self.model = model
        self.eps = eps
        self.trained_options = []
        ob = U.get_placeholder(name="ob", dtype=tf1.float32, shape=[sequence_length] + list(ob_space.shape))
        option = U.get_placeholder(name="option", dtype=tf1.int32, shape=[None])
        self.pdtype = pdtype = DiagGaussianPdType(ac_space.shape[0])
        with tf1.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf1.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz

        # Value function
        for i in range(num_hid_layers[0]):
            last_out = tf1.nn.tanh(tf1.layers.dense(last_out, hid_size[0], name="vffc%i" % (i + 1),
                                                    kernel_initializer=U.normc_initializer(1.0)))
        self.vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:, 0]

        # Intra option policy
        last_out = ob
        for i in range(num_hid_layers[1]):
            last_out = tf1.nn.tanh(tf1.layers.dense(last_out, hid_size[1], name="polfc%i" % (i + 1),
                                                    kernel_initializer=U.normc_initializer(1.0)))

        mean = dense3D2(last_out, pdtype.param_shape()[0] // 2, "polfinal", option, num_options=num_options,
                        weight_init=U.normc_initializer(-0.75))
        logstd = tf1.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0] // 2], initializer=U.normc_initializer(0.5), trainable=True)
        pdparam = tf1.concat([mean, mean * 0.0 + logstd[option[0]]], axis=1)

        # pdparam = dense3D2(last_out, pdtype.param_shape()[0], "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(-0.6))
        self.pd = pdtype.pdfromflat(pdparam)
        stochastic = tf1.placeholder(dtype=tf1.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic, ob, option], [ac])
        self.get_vpred = U.function([ob, option], [self.vpred])
        self.action_pd = U.function([ob, option], [self.pd.mode(), self.pd.variance()])

    def act(self, stochastic, ob, option):
        ac1 = self._act(stochastic, ob[None], [option])
        return ac1[0][0]

    def get_ac_dist(self, ob, option):
        mean, std = self.action_pd(ob[None], [option])
        return mean[0], std[0]

    def get_intfc(self, ob, constraint):
        return self.model.getInterest(ob, constraint)

    def get_tpred(self, ob):
        return self.model.getTermination(ob)

    def get_preds(self, ob, constraint):
        beta = self.get_tpred(ob)
        int_func = self.get_intfc(ob, constraint)
        # Get Q(s,w)
        vpred = []
        for opt in range(self.num_options):
            vpred.append(self.get_vpred(ob, [opt])[0])
        vpred = np.array(vpred).T
        # Get V(s)
        op_vpred = np.sum((int_func * vpred), axis=1)

        return beta, vpred, op_vpred

    def get_option(self, ob, constraint):
        int_func = self.get_intfc(ob, constraint)
        activated_options = int_func
        vpred = []
        # max Q(s,w)
        for opt in range(self.num_options):
            vpred.append(int_func[opt] * self.get_vpred(ob, [opt])[0])
        vpred = np.array(vpred)
        max = float('-inf')
        available_options = []
        for opt in range(self.num_options):
            if int_func[opt] == 1:
                available_options.append(opt)
                if vpred[opt][0] > max:
                    option = opt
                    max = vpred[opt][0]
        p = np.random.random()
        if p < self.eps:
            option = random.choice(available_options)

        return option, activated_options

    def get_variables(self):
        return tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
