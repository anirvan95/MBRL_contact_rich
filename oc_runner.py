import tensorflow.compat.v1 as tf
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

tf.disable_v2_behavior()


def sample_trajectory(pi, env, horizon=150, rolloutSize=50, render=False):
    if render:
        env.setRender(True)
    else:
        env.setRender(False)
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    num_options = pi.num_options
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    batch_size = int(horizon * rolloutSize)
    # Initialize history arrays
    obs = np.array([ob for _ in range(batch_size)])
    rews = np.zeros(batch_size, 'float32')
    news = np.zeros(batch_size, 'int32')
    opts = np.zeros(batch_size, 'int32')
    activated_options = np.zeros((batch_size, num_options), 'float32')

    last_options = np.zeros(batch_size, 'int32')
    acs = np.array([ac for _ in range(batch_size)])
    prevacs = acs.copy()

    option, active_options_t = pi.get_option(ob)

    option_plots = []
    option_terms = []
    int_vals = []
    option_plots.append(option)
    last_option = option

    term_prob = pi.get_tpred([ob], [option])[0][0]

    option_terms.append(term_prob)
    int_val = pi.get_int_func([ob])
    int_vals.append(int_val[0, option])

    ep_states = [[] for _ in range(num_options)]
    ep_states[option].append(ob)
    ep_num = 0

    opt_duration = [[] for _ in range(num_options)]
    sample_index = 0
    curr_opt_duration = 0.

    success = 0
    successFlag = False

    while sample_index < batch_size:
        prevac = ac
        ac = pi.act(True, ob, option)
        obs[sample_index] = ob
        last_options[sample_index] = last_option
        news[sample_index] = new
        opts[sample_index] = option
        acs[sample_index] = ac
        prevacs[sample_index] = prevac
        activated_options[sample_index] = active_options_t

        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()
            time.sleep(0.0005)
        rews[sample_index] = rew

        curr_opt_duration += 1
        term = pi.get_term([ob], [option])[0][0]

        candidate_option, active_options_t = pi.get_option(ob)
        if term:
            if num_options > 1:
                rews[sample_index] -= pi.dc
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            option = candidate_option
            term_prob = pi.get_tpred([ob], [option])[0][0]

        option_terms.append(term_prob)

        ep_states[option].append(ob)

        cur_ep_ret += rew
        cur_ep_len += 1

        dist = env.getGoalDist()
        if np.linalg.norm(dist) < 0.02 and not successFlag:
            success = success + 1
            successFlag = True

        sample_index += 1

        if new or (sample_index > 0 and sample_index % horizon == 0):
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
            new = True
            successFlag = False
            render = False
            env.setRender(False)
            term_prob = pi.get_tpred([ob], [option])[0][0]

    vpreds, op_vpreds, vpred, op_vpred, op_probs, intfc = pi.get_allvpreds(obs, ob)
    term_ps, term_p = pi.get_alltpreds(obs, ob)
    last_betas = term_ps[range(len(last_options)), last_options]
    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    rollouts = {"ob": obs, "rew": rews, "vpred": vpreds, "op_vpred": op_vpreds, "new": news,
           "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new),
           "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': term_ps, 'next_term_p': term_p,
           "opt_dur": opt_duration, "op_probs": op_probs, "last_betas": last_betas, "intfc": intfc,
           "activated_options": activated_options, "success": success}

    return rollouts


def add_vtarg_and_adv(rollouts, gamma, lam, num_options):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(rollouts["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    op_vpred = np.append(rollouts["op_vpred"], rollouts["nextop_vpred"])
    T = len(rollouts["rew"])
    rollouts["op_adv"] = gaelam = np.empty(T, 'float32')
    rew = rollouts["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * op_vpred[t + 1] * nonterminal - op_vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    rollouts["op_adv"] = gaelam

    term_p = np.vstack((np.array(rollouts["term_p"]), np.array(rollouts["next_term_p"])))
    q_sw = np.vstack((rollouts["vpred"], rollouts["nextvpred"]))
    u_sw = (1 - term_p) * q_sw + term_p * np.tile(op_vpred[:, None], num_options)
    opts = rollouts["opts"]

    new = np.append(rollouts["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    T = len(rollouts["rew"])
    rew = rollouts["rew"]
    gaelam = np.empty((num_options, T), 'float32')
    for opt in range(num_options):
        vpred = u_sw[:, opt]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[opt, t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    rollouts["adv"] = gaelam.T[range(len(opts)), opts]
    rollouts["tdlamret"] = rollouts["adv"] + u_sw[range(len(opts)), opts]


def learn(env, model_path, data_path, policy_fn, *,
          rolloutSize, num_options=4, horizon=80,
          clip_param=0.025, ent_coeff=0.01,  # clipping parameter epsilon, entropy coeff
          optim_epochs=10, mainlr=3.25e-4, intlr=1e-4, piolr=1e-4, termlr=5e-7, optim_batchsize=100,  # optimization hypers
          gamma=0.99, lam=0.95,  # advantage estimation
          max_iters=20,  # time constraint
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          retrain=False,
          ):
    """
        Core learning function
    """
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space, num_options=num_options)  # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space, num_options=num_options)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])
    op_adv = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    betas = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    # Setup losses and stuff
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-ent_coeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    term_loss = pi.tpred * term_adv

    activated_options = tf.placeholder(dtype=tf.float32, shape=[None, num_options])
    pi_w = tf.placeholder(dtype=tf.float32, shape=[None, num_options])
    option_hot = tf.one_hot(option, depth=num_options)
    pi_I = (pi.intfc * activated_options) * pi_w / tf.expand_dims(
        tf.reduce_sum((pi.intfc * activated_options) * pi_w, axis=1), 1)
    pi_I = tf.clip_by_value(pi_I, 1e-6, 1 - 1e-6)
    int_loss = - tf.reduce_sum(betas * tf.reduce_sum(pi_I * option_hot, axis=1) * op_adv)

    intfc = tf.placeholder(dtype=tf.float32, shape=[None, num_options])
    pi_I = (intfc * activated_options) * pi.op_pi / tf.expand_dims(
        tf.reduce_sum((intfc * activated_options) * pi.op_pi, axis=1), 1)
    pi_I = tf.clip_by_value(pi_I, 1e-6, 1 - 1e-6)
    op_loss = - tf.reduce_sum(betas * tf.reduce_sum(pi_I * option_hot, axis=1) * op_adv)

    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-20, 1.0))
    op_entropy = -tf.reduce_mean(pi.op_pi * log_pi, reduction_indices=1)
    op_loss -= 0.01 * tf.reduce_sum(op_entropy)

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option], losses + [U.flatgrad(total_loss, var_list)])
    termgrad = U.function([ob, option, term_adv],
                          [U.flatgrad(term_loss, var_list)])  # Since we will use a different step size.
    opgrad = U.function([ob, option, betas, op_adv, intfc, activated_options],
                        [U.flatgrad(op_loss, var_list)])  # Since we will use a different step size.
    intgrad = U.function([ob, option, betas, op_adv, pi_w, activated_options],
                         [U.flatgrad(int_loss, var_list)])  # Since we will use a different step size.
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)

    U.initialize()
    adam.sync()

    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=5)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=5)  # rolling buffer for episode rewards

    datas = [0 for _ in range(num_options)]

    if retrain:
        print("Retraining to New Task !! ")
        time.sleep(2)
        U.load_state(model_path+'/')

    p = []
    max_timesteps = int(horizon * rolloutSize * max_iters)
    while True:
        if max_iters and iters_so_far >= max_iters:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)
        render = False

        rollouts = sample_trajectory(pi, env, horizon=horizon, rolloutSize=rolloutSize, render=render)
        # Save rollouts
        data = {'rollouts': rollouts}
        p.append(data)
        del data
        data_file_name = data_path + 'rollout_data.pkl'
        pickle.dump(p, open(data_file_name, "wb"))

        add_vtarg_and_adv(rollouts, gamma, lam, num_options)

        opt_d = []
        for i in range(num_options):
            dur = np.mean(rollouts['opt_dur'][i]) if len(rollouts['opt_dur'][i]) > 0 else 0.
            opt_d.append(dur)

        ob, ac, opts, atarg, tdlamret = rollouts["ob"], rollouts["ac"], rollouts["opts"], rollouts["adv"], rollouts["tdlamret"]
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy
        assign_old_eq_new()  # set old parameter values to new parameter values

        # Optimizing the policy
        for opt in range(num_options):
            indices = np.where(opts == opt)[0]
            print("Option- ", opt, " Batch Size: ", indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                continue

            datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)

            if indices.size < optim_batchsize:
                print("Too few samples for opt - ", opt)
                continue

            optim_batchsize_corrected = optim_batchsize
            optim_epochs_corrected = np.clip(np.int(indices.size / optim_batchsize_corrected), 1, optim_epochs)
            print("Optim Epochs:", optim_epochs_corrected)
            logger.log("Optimizing...")
            # Here we do a bunch of optimization epochs over the data

            for _ in range(optim_epochs_corrected):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize_corrected):
                    *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                    cur_lrmult, [opt])
                    adam.update(grads, mainlr * cur_lrmult)
                    losses.append(newlosses)

            # Optimize termination functions
            termg = termgrad(rollouts["ob"], rollouts['opts'], rollouts["op_adv"])[0]
            adam.update(termg, termlr)

            # Optimize interest functions
            intgrads = intgrad(rollouts['ob'], rollouts['opts'], rollouts["last_betas"], rollouts["op_adv"], rollouts["op_probs"], rollouts["activated_options"])[0]
            adam.update(intgrads, intlr)

        # Optimize policy over options
        opgrads = opgrad(rollouts['ob'], rollouts['opts'], rollouts["last_betas"], rollouts["op_adv"], rollouts["intfc"], rollouts["activated_options"])[0]
        adam.update(opgrads, piolr)

        lrlocal = (rollouts["ep_lens"], rollouts["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("Success", rollouts["success"])
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

    return pi


