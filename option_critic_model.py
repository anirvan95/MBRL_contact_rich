import tensorflow.compat.v1 as tf1
import common.tf_util as U
import numpy as np
from common.distribution import DiagGaussianPdType
import gym
from common.mpi_running_mean_std import RunningMeanStd
from common.model_learning_utils import LRPrediction
import time

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

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, num_options=2, dc=0, w_intfc=True, k=0.,):
        assert isinstance(ob_space, gym.spaces.Box)
        self.k = k
        self.w_intfc = w_intfc
        self.state_in = []
        self.state_out = []
        self.dc = dc
        self.num_options = num_options
        self.nmodes = 1
        # Creating the policy network
        sequence_length = None
        self.ac_dim = ac_space.shape[0]

        ob = U.get_placeholder(name="ob", dtype=tf1.float32, shape=[sequence_length] + list(ob_space.shape))
        option = U.get_placeholder(name="option", dtype=tf1.int32, shape=[None])

        self.pdtype = pdtype = DiagGaussianPdType(ac_space.shape[0])


        with tf1.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf1.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        last_out = obz
        # Value function
        for i in range(num_hid_layers):
            last_out = tf1.nn.tanh(tf1.layers.dense(last_out, hid_size, name="vffc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:, 0]

        # Intra option policy
        last_out = ob
        for i in range(num_hid_layers):
            last_out = tf1.nn.tanh(
                tf1.layers.dense(last_out, hid_size, name="polfc%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))

        mean = dense3D2(last_out, pdtype.param_shape()[0] // 2, "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.01))
        logstd = tf1.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0] // 2], initializer=tf1.zeros_initializer())
        pdparam = tf1.concat([mean, mean * 0.0 + logstd[option[0]]], axis=1)
        self.pd = pdtype.pdfromflat(pdparam)
        stochastic = tf1.placeholder(dtype=tf1.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        # Option policy
        last_out = ob
        for i in range(num_hid_layers):
            last_out = tf1.nn.tanh(
                tf1.layers.dense(last_out, hid_size, name="OP%i" % (i + 1), kernel_initializer=U.normc_initializer(1.0)))
        self.op_pi = tf1.nn.softmax(tf1.layers.dense(last_out, num_options, name="OPfinal", kernel_initializer=U.normc_initializer(1.0)))

        self._act = U.function([stochastic, ob, option], [ac])
        self.get_vpred = U.function([ob, option], [self.vpred])
        self._get_op = U.function([ob], [self.op_pi])
        #Try to add limit to ac generated ac, bounding box

    def act(self, stochastic, ob, option):

        ac1 = self._act(stochastic, ob[None], [option])
        return ac1[0]

    def init_hybridmodel(self, lr_params_interest, lr_params_guard):
        self.intfc = LRPrediction(lr_params_interest)
        self.termfc = LRPrediction(lr_params_guard)

    def learn_hybridmodel(self, intXD, intYD, termYD):
        for datasets in range(0, len(intXD)):
            if datasets == 0:
                intX = intXD[datasets]
                intY = intYD[datasets]
                termY = termYD[datasets]
            else:
                intX = np.vstack((intX, intXD[datasets]))
                intY = np.vstack((intY, intYD[datasets]))
                termY = np.vstack((termY, termYD[datasets]))

        self.intfc.train(intX, np.squeeze(intY))
        self.termfc.train(intX, np.squeeze(termY))
    '''
    def get_tpred(self, ob, option):
        mc_samples = 24
        if self.nmodes == self.num_options:
            ac_dist = []
            for i in range(0, mc_samples):
                ac_dist.append(self.act(True, ob, option))
            ac_dist = np.array(ac_dist).reshape(mc_samples, self.ac_dim)
            ob = np.expand_dims(ob, axis=0)
            ob_dist = np.repeat(ob, mc_samples, axis=0)
            ob_ac = np.hstack((ob_dist, ac_dist))
            noption = self.termfc.predict(ob_ac)
            t_pred = 1 - sum(1 for op in noption if op == option)/mc_samples
        else:
            t_pred = 0.0
        return t_pred
    '''

    # TODO: Generalize to num_options > 2

    def get_intfc(self, obs):
        if self.nmodes == self.num_options:
            int_fc = self.intfc.predict_f(obs)
            #include assert to check prob
        else:
            int_fc = np.array([[1.0, 0.0]])
        return int_fc

    def get_tpred(self, obs):
        if self.nmodes == self.num_options:
            beta = np.array([[1.0, 1.0]]) - self.termfc.predict_f(obs)
            #include assert to check prob
        else:
            beta = np.array([[0.0, 1.0]])
        return beta

    def get_preds(self, ob):
        # Get B(s,w)
        '''
        beta = []
        for opt in range(self.num_options):
            beta.append(self.get_tpred(ob, opt))
        beta = np.array(beta).T
        '''
        beta = self.get_tpred([ob])
        # Get V(s,w)
        vpred = []
        for opt in range(self.num_options):
            vpred.append(self.get_vpred(ob, [opt])[0])
        vpred = np.array(vpred).T
        # Get V(w)
        op_prob = self._get_op(ob)
        int_func = self.get_intfc([ob])
        op_prob = op_prob[0]
        pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
        op_vpred = np.sum((pi_I * vpred), axis=1)  # Get V(s)

        return beta, vpred, op_vpred, op_prob, int_func
    '''
    def get_allvpreds(self, obs, ob):
        obs = np.vstack((obs, ob[None]))

        # Get Q(s,w)
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs, [opt])[0])
        vals = np.array(vals).T

        op_prob = self._get_op(obs)
        int_func = self.get_intfc(obs)
        op_prob = op_prob[0]

        pi_I = op_prob * int_func / np.sum(op_prob * int_func, axis=1)[:, None]
        op_vpred = np.sum((pi_I * vals), axis=1)  # Get V(s)

        return vals[:-1], op_vpred[:-1], vals[-1], op_vpred[-1], op_prob[:-1], int_func[:-1]

    def get_vpreds(self, obs):
        vals = []
        for opt in range(self.num_options):
            vals.append(self.get_vpred(obs, [opt])[0])
        vals = np.array(vals).T
        return vals
    '''
    def get_option(self, ob):

        op_prob = self._get_op([ob])
        int_func = self.get_intfc([ob])

        activated_options = []

        for int_val in int_func[0]:
            if int_val >= 0.5:
                activated_options.append(1.)
            else:
                activated_options.append(0.)
        indices = (-int_func[0]).argsort()[:2]
        if 1. not in activated_options:
            for i in indices:
                activated_options[i] = 1.

        pi_I = op_prob[0] * (activated_options * int_func) / np.sum(op_prob[0] * (activated_options * int_func), axis=1)

        return np.random.choice(range(len(op_prob[0][0])), p=pi_I[0]), activated_options

    def get_variables(self):
        return tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []