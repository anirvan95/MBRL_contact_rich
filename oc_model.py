from common.mpi_running_mean_std import RunningMeanStd
import common.tf_util as U
import tensorflow.compat.v1 as tf
import gym
from common.distribution import DiagGaussianPdType
import numpy as np


def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w[option[0]])
    if bias:
        b = tf.get_variable(name + "/b", [num_options, size], initializer=tf.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, num_options=2, dc=0, w_intfc=True, k=0.):
        assert isinstance(ob_space, gym.spaces.Box)
        self.k = k
        self.w_intfc = w_intfc
        self.state_in = []
        self.state_out = []
        self.dc = dc
        self.num_options = num_options
        self.pdtype = pdtype = pdtype = DiagGaussianPdType(ac_space.shape[0])
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        option = U.get_placeholder(name="option", dtype=tf.int32, shape=[None])

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="vffc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="termfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.tpred = tf.nn.sigmoid(dense3D2(tf.stop_gradient(last_out), 1, "termhead", option, num_options=num_options, weight_init=U.normc_initializer(1.0)))[:, 0]
        termination_sample = tf.greater(self.tpred, tf.random_uniform(shape=tf.shape(self.tpred), maxval=1.))

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="polfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense3D2(last_out, pdtype.param_shape()[0] // 2, "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.5))
            logstd = tf.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0] // 2], initializer=U.normc_initializer(0.1), trainable=True)
            pdparam = tf.concat([mean, mean * 0.0 + logstd[option[0]]], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="intfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.intfc = tf.sigmoid(tf.layers.dense(last_out, num_options, name="intfcfinal", kernel_initializer=U.normc_initializer(1.0)))

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="OP%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.op_pi = tf.nn.softmax(tf.layers.dense(last_out, num_options, name="OPfinal", kernel_initializer=U.normc_initializer(1.0)))

        self._act = U.function([stochastic, ob, option], [ac])
        self.get_term = U.function([ob, option], [termination_sample])
        self.get_tpred = U.function([ob, option], [self.tpred])
        self.get_vpred = U.function([ob, option], [self.vpred])
        self._get_op_int = U.function([ob], [self.op_pi, self.intfc])
        self._get_intfc = U.function([ob], [self.intfc])
        self._get_op = U.function([ob], [self.op_pi])

    def act(self, stochastic, ob, option):
        ac1 = self._act(stochastic, ob[None], [option])
        return ac1[0][0]

    def get_int_func(self, obs):
        return self._get_intfc(obs)[0]

    def get_alltpreds(self, obs, ob):
        obs = np.vstack((obs, ob[None]))

        # Get B(s,w)
        betas = []
        for opt in range(self.num_options):
            betas.append(self.get_tpred(obs, [opt])[0])
        betas = np.array(betas).T

        return betas[:-1], betas[-1]

    def get_allvpreds(self, obs, ob):
        obs = np.vstack((obs, ob[None]))

        # Get Q(s,w)
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs, [opt])[0])
        vals = np.array(vals).T

        op_prob, int_func = self._get_op_int(obs)
        if self.w_intfc:
            pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
        else:
            pi_I = op_prob
        op_vpred = np.sum((pi_I * vals), axis=1)  # Get V(s)

        return vals[:-1], op_vpred[:-1], vals[-1], op_vpred[-1], op_prob[:-1], int_func[:-1]

    def get_vpreds(self, obs):
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs, [opt])[0])
        vals = np.array(vals).T
        return vals

    def get_option(self, ob):

        op_prob, int_func = self._get_op_int([ob])
        activated_options = []
        for int_val in int_func[0]:
            if int_val >= self.k:
                activated_options.append(1.)
            else:
                activated_options.append(0.)
        indices = (-int_func[0]).argsort()[:2]
        if 1. not in activated_options:
            for i in indices:
                activated_options[i] = 1.

        if self.w_intfc:
            pi_I = op_prob * (activated_options * int_func) / np.sum(op_prob * (activated_options * int_func), axis=1)[
                                                              :, None]
        else:
            pi_I = op_prob

        return np.random.choice(range(len(op_prob[0])), p=pi_I[0]), activated_options

    def get_intvals(self, ob):
        op_prob, int_func = self._get_op_int([ob])
        return op_prob, int_func

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []