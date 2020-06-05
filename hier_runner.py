import tensorflow.compat.v1 as tf1
from common.clustering_utils import hybridSegmentClustering
from common.dataset import Dataset
import logger
import common.tf_util as U
import numpy as np
from common.mpi_adam import MpiAdam
from mpi4py import MPI
from collections import deque
from common.math_util import zipsame, flatten_lists
import time
import pickle
import math

tf1.disable_v2_behavior()


'''
# collect trajectory for model learning
def sample_trajectory_model_learning(pi, env, horizon=150, batch_size=3000):
    """
                Generates rollouts for model_learning
    """
    n_samples = 0
    ob = env.reset()
    ac = env.action_space.sample()
    observations = np.array([ob for _ in range(horizon)])
    actions = np.array([ac for _ in range(horizon)])
    option, active_options_t = pi.get_option(ob)
    t = 0
    rollouts = []
    while n_samples < batch_size:
        ac = pi.act(True, ob, option)
        observations[t] = ob
        actions[t] = ac
        ob, rew, new, _ = env.step(ac)
        beta = pi.get_tpred([ob])
        tprob = beta[0][option]
        if tprob > pi.term_prob:
            option, active_options_t = pi.get_option(ob)
        t = t + 1
        if t == horizon or new:
            data = {'observations': observations, 'actions': actions}
            rollouts.append(data)
            ob = env.reset()
            option, active_options_t = pi.get_option(ob)
            t = 0

        n_samples = n_samples + 1

    return rollouts
'''

# collect trajectory
def sample_trajectory(pi, env, horizon=150, batch_size=12000, render=False):
    """
            Generates rollouts for policy optimization
    """
    GOAL = np.array([0, 0.5])
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    num_options = pi.num_options
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialise history of arrays
    obs = np.array([ob for _ in range(batch_size)])
    rews = np.zeros(batch_size, 'float32')
    news = np.zeros(batch_size, 'int32')
    opts = np.zeros(batch_size, 'int32')
    activated_options = np.zeros((batch_size, num_options), 'float32')

    last_options = np.zeros(batch_size, 'int32')
    acs = np.array([ac for _ in range(batch_size)])
    prevacs = acs.copy()
    option, active_options_t = pi.get_option(ob)
    last_option = option

    betas = []
    vpreds = []
    op_vpreds = []
    int_fcs = []
    op_probs = []

    ep_states = [[] for _ in range(num_options)]
    ep_states[option].append(ob)
    ep_num = 0

    opt_duration = [[] for _ in range(num_options)]
    t = 0
    curr_opt_duration = 0

    insertion = 0
    new_rollout = True
    while t < batch_size:
        prevac = ac
        ac = pi.act(True, ob, option)
        obs[t] = ob
        last_options[t] = last_option
        news[t] = new
        opts[t] = option
        acs[t] = ac
        prevacs[t] = prevac
        beta, vpred, op_vpred, op_prob, int_fc = pi.get_preds(ob)

        betas.append(beta[0])
        vpreds.append(vpred*(1-new))
        op_vpreds.append(op_vpred)
        int_fcs.append(int_fc)
        op_probs.append(op_prob)
        activated_options[t] = active_options_t

        ob, rew, new, _ = env.step(ac)
        if render:
            #print("Beta function :", beta)
            #print("Interest function: ",int_fc)
            #print("Current option :", option)
            env.render()
            #time.sleep(0.5)

        rews[t] = rew
        curr_opt_duration += 1
        # check if current option is about to end in this state
        nbeta = pi.get_tpred(ob)
        tprob = nbeta[0][option]

        if tprob >= pi.term_prob:
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            option, active_options_t = pi.get_option(ob)

        cur_ep_ret += rew
        cur_ep_len += 1
        dist = ob[:2] - GOAL
        if np.linalg.norm(dist) < 0.025 and new_rollout:
            insertion = insertion + 1
            new_rollout = False

        if new or (t > 0 and t % horizon == 0):
            render = False
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ep_num += 1
            ob = env.reset()
            option, active_options_t = pi.get_option(ob)
            last_option = option
            new_rollout = True
            new = True
        t += 1

    betas = np.array(betas)
    vpreds = np.array(vpreds).reshape(batch_size, num_options)
    op_vpreds = np.squeeze(np.array(op_vpreds))
    op_probs = np.array(op_probs).reshape(batch_size, num_options)
    int_fcs = np.array(int_fcs).reshape(batch_size, num_options)
    last_betas = betas[range(len(last_options)), last_options]
    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg = {"ob": obs, "rew": rews, "vpred": np.array(vpreds), "op_vpred": np.array(op_vpreds), "new": news,
           "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new),
           "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': betas, 'next_term_p': beta[0],
           "opt_dur": opt_duration, "op_probs": np.array(op_probs), "last_betas": last_betas,
           "intfc": np.array(int_fcs),
           "activated_options": activated_options, "success": insertion}

    return seg


def add_vtarg_and_adv(seg, gamma, lam, num_options):
    """
        Compute advantage and other value functions using GAE
    """
    new = np.append(seg["new"], True)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    T = len(seg["rew"])
    gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0

    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        # TD error
        delta = rew[t] + gamma * op_vpred[t + 1] * nonterminal - op_vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["op_adv"] = gaelam

    term_p = np.vstack((np.array(seg["term_p"]), np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"], seg["nextvpred"]))
    # Utility function in option framework
    u_sw = (1 - term_p) * q_sw + term_p * np.tile(op_vpred[:, None], num_options)

    opts = seg["opts"]
    new = np.append(seg["new"], 0)
    T = len(seg["rew"])
    rew = seg["rew"]
    gaelam = np.empty((num_options, T), 'float32')
    for opt in range(num_options):
        vpred = u_sw[:, opt]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[opt, t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["adv"] = gaelam.T[range(len(opts)), opts]
    seg["tdlamret"] = seg["adv"] + u_sw[range(len(opts)), opts]


def learn(env, policy_fn, clustering_params, lr_params_interest, lr_params_guard, *, num_options=2,
          horizon,  # timesteps per actor per update
          clip_param, pol_entcoeff=0.02, op_entcoeff=0.01, # clipping parameter epsilon, entropy coeff
          optim_epochs=10, mainlr=3e-4, intlr=1e-4, optim_batchsize=160,  # optimization hypers
          gamma=0.99, lam=0.95,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          batch_size_per_episode=15000,
          adam_epsilon=1.2e-4,
          schedule='linear'  # annealing for stepsize parameters (epsilon and adam)
          ):
    """
            Core learning function
    """

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    #pi.init_hybridmodel(lr_params_interest, lr_params_guard)
    oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Empirical return

    lrmult = tf1.placeholder(name='lrmult', dtype=tf1.float32,
                             shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    op_adv = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Target advantage function (if applicable)
    betas = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    # Defining losses for optimization
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf1.reduce_mean(kloldnew)
    meanent = tf1.reduce_mean(ent)
    pol_entpen = (-pol_entcoeff) * meanent

    ratio = tf1.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf1.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf1.reduce_mean(tf1.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = tf1.reduce_mean(tf1.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    activated_options = tf1.placeholder(dtype=tf1.float32, shape=[None, num_options])
    option_hot = tf1.one_hot(option, depth=num_options)
    intfc = tf1.placeholder(dtype=tf1.float32, shape=[None, num_options])
    pi_I = (intfc * activated_options) * pi.op_pi / tf1.expand_dims(tf1.reduce_sum((intfc * activated_options) * pi.op_pi, axis=1), 1)
    pi_I = tf1.clip_by_value(pi_I, 1e-6, 1 - 1e-6)
    op_loss = - tf1.reduce_sum(betas * tf1.reduce_sum(pi_I * option_hot, axis=1) * op_adv)
    log_op_pi = tf1.log(tf1.clip_by_value(pi.op_pi, 1e-20, 1.0))
    op_entropy = -tf1.reduce_mean(pi.op_pi * log_op_pi, reduction_indices=1)
    op_loss -= op_entcoeff * tf1.reduce_sum(op_entropy)

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option], losses + [U.flatgrad(total_loss, var_list)])
    lossandgrad_vf = U.function([ob, ac, atarg, ret, lrmult, option], losses + [U.flatgrad(vf_loss, var_list)])
    opgrad = U.function([ob, option, betas, op_adv, intfc, activated_options], [U.flatgrad(op_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf1.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)

    U.initialize()
    adam.sync()

    currIter = 0
    optim_stepsize = mainlr
    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=5)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=5)  # rolling buffer for episode rewards

    # rolling buffers for training interest and guard functions
    label_train_dataset = deque(maxlen=3)
    label_t_train_dataset = deque(maxlen=3)
    xu_train_dataset = deque(maxlen=3)
    x_train_dataset = deque(maxlen=3)
    #pickel variable
    p = []

    model_learning_flag = True
    retain_model = False

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)
        print("Collecting samples for policy optimization !! ")
        stime = time.time()
        render = False

        seg = sample_trajectory(pi, env, horizon=horizon, batch_size=batch_size_per_episode, render=render)
        print("Samples collected in !! :", time.time() - stime)

        datas = [0 for _ in range(num_options)]
        add_vtarg_and_adv(seg, gamma, lam, num_options)

        opt_d = []
        for i in range(num_options):
            dur = np.mean(seg['opt_dur'][i]) if len(seg['opt_dur'][i]) > 0 else 0.
            opt_d.append(dur)

        ob, ac, opts, atarg, tdlamret = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"]

        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(ob)  # update running mean/std for policy
        assign_old_eq_new()

        for opt in range(num_options):
            indices = np.where(opts == opt)[0]
            print("Batch Size:", indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                continue

            datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)

            if indices.size < optim_batchsize:
                print("Too few samples")
                continue

            optim_batchsize_corrected = optim_batchsize
            optim_epochs_corrected = np.clip(np.int(indices.size / optim_batchsize_corrected), 1,  optim_epochs)
            print("Optim Epochs:", optim_epochs_corrected)
            logger.log("Optimizing...")
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs_corrected):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize_corrected):
                    *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt])
                    adam.update(grads, mainlr * cur_lrmult)
                    losses.append(newlosses)

        opgrads = opgrad(seg['ob'], seg['opts'], seg["last_betas"], seg["op_adv"], seg["intfc"], seg["activated_options"])[0]
        adam.update(opgrads, intlr)
        '''
        # Update model with updated policy
        rollouts = sample_trajectory_model_learning(pi, env, horizon=horizon, batch_size=int(batch_size_per_episode / 4))
        print("Updating Model")
        nmodes, segmentedRollouts, x_train, u_train, delx_train, label_train, label_t_train = hybridSegmentClustering(rollouts, num_options, clustering_params)
        data = {'seg': seg, 'rollouts': rollouts}
        p.append(data)
        pickle.dump(p, open("data/option_critic_data_exp_8.pkl", "wb"))

        if seg["success"] > 10 and model_learning_flag:
            model_learning_flag = False
            retain_model = True

        if retain_model:
            print("Going to retain model")
            pi.nmodes = num_options
        else:
            pi.nmodes = nmodes

        if nmodes == num_options:  # slight hack here, currently only testing with 2 modes, improve to nmodes
            # xu_train = np.hstack((x_train, u_train))
            x_train_dataset.append(x_train)
            label_train_dataset.append(label_train)
            label_t_train_dataset.append(label_t_train)
            # xu_train_dataset.append(xu_train)
            pi.learn_hybridmodel(x_train_dataset, label_train_dataset, x_train_dataset, label_t_train_dataset)
        '''
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("SuccessInsertion", seg["success"])
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

