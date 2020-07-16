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
        ob = U.get_placeholder(name="ob", dtype=tf1.float32, shape=[sequence_length] + list(ob_space.shape))
        option = U.get_placeholder(name="option", dtype=tf1.int32, shape=[None])
        self.pdtype = pdtype = DiagGaussianPdType(ac_space.shape[0])
        with tf1.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf1.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz

        # Value function
        for i in range(num_hid_layers[2]):
            last_out = tf1.nn.tanh(tf1.layers.dense(last_out, hid_size[2], name="vffc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:, 0]

        # Intra option policy
        last_out = ob
        for i in range(num_hid_layers[1]):
            last_out = tf1.nn.tanh(
                tf1.layers.dense(last_out, hid_size[1], name="polfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))

        mean = dense3D2(last_out, pdtype.param_shape()[0] // 2, "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.01))
        logstd = tf1.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0] // 2], initializer=tf1.zeros_initializer())
        pdparam = tf1.concat([mean, mean * 0.0 + logstd[option[0]]], axis=1)
        self.pd = pdtype.pdfromflat(pdparam)
        stochastic = tf1.placeholder(dtype=tf1.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        # Option policy
        last_out = ob
        for i in range(num_hid_layers[0]):
            last_out = tf1.nn.tanh(tf1.layers.dense(last_out, hid_size[0], name="OP%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.op_pi = tf1.nn.softmax(tf1.layers.dense(last_out, num_options, name="OPfinal", kernel_initializer=U.normc_initializer(1.0)))

        self._act = U.function([stochastic, ob, option], [ac])
        self.get_vpred = U.function([ob, option], [self.vpred])
        self._get_op = U.function([ob], [self.op_pi])

    def act(self, stochastic, ob, option):
        ac1 = self._act(stochastic, ob[None], [option])
        return ac1[0][0]

    def get_intfc(self, ob):
        return self.model.getInterest(ob)

    def get_tpred(self, ob):
        return self.model.getTermination(ob)

    def get_preds(self, ob):
        beta = self.get_tpred(ob)
        int_func = self.get_intfc(ob)
        # Get Q(s,w)
        vpred = []
        for opt in range(self.num_options):
            vpred.append(self.get_vpred(ob, [opt])[0])
        vpred = np.array(vpred).T
        # Get V(s)
        op_vpred = np.sum((int_func * vpred), axis=1)

        return beta, vpred, op_vpred

    def get_preds_adv(self, ob):
        # Get B(s,w)
        '''
        beta = []
        for opt in range(self.num_options):
            beta.append(self.get_tpred(ob, opt))
        beta = np.array(beta).T
        '''
        beta = self.get_tpred(ob)
        # Get V(s,w)
        vpred = []
        for opt in range(self.num_options):
            vpred.append(self.get_vpred(ob, [opt])[0])
        vpred = np.array(vpred).T
        # Get V(w)
        op_prob = self._get_op(ob)
        int_func = self.get_intfc(ob)
        op_prob = op_prob[0]
        pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
        op_vpred = np.sum((pi_I * vpred), axis=1)  # Get V(s)

        return beta, vpred, op_vpred, op_prob, int_func

    def get_option(self, ob):
        int_func = self.get_intfc(ob)
        activated_options = int_func
        vpred = []
        # max Q(s,w)
        for opt in range(self.num_options):
            vpred.append(int_func[opt]*self.get_vpred(ob, [opt])[0])
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

    def get_option_adv(self, ob):

        op_prob = self._get_op([ob])
        int_func = self.get_intfc(ob)

        activated_options = []
        # Include option if the interest is high but option policy does not select it
        for int_val in int_func[0]:
            if int_val >= 0.5:
                activated_options.append(1.)
            else:
                activated_options.append(0.)
        indices = (-int_func[0]).argsort()[:2]
        if 1. not in activated_options:
            for i in indices:
                activated_options[i] = 1.
        try:
            pi_I = op_prob[0] * (activated_options * int_func) / np.sum(op_prob[0] * (activated_options * int_func), axis=1)
        except ValueError:
            print("Value error in option selection")

        return np.random.choice(range(len(op_prob[0][0])), p=pi_I[0]), activated_options

    def get_variables(self):
        return tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
