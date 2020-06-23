import tensorflow.compat.v1 as tf
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

tf.disable_v2_behavior()


def traj_segment_generator(pi, env, batchSize, horizon, stochastic, num_options, dc):
    GOAL = np.array([0, 0.52])
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    t = 0
    first_ep = True

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(batchSize)])
    rews = np.zeros(batchSize, 'float32')
    realrews = np.zeros(batchSize, 'float32')
    news = np.zeros(batchSize, 'int32')
    opts = np.zeros(batchSize, 'int32')
    activated_options = np.zeros((batchSize, num_options), 'float32')

    last_options = np.zeros(batchSize, 'int32')
    acs = np.array([ac for _ in range(batchSize)])
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

    curr_opt_duration = 0.

    insertion = 0
    new_rollout = True

    while t < batchSize:
        prevac = ac
        ac = pi.act(stochastic, ob, option)
        obs[t] = ob
        last_options[t] = last_option
        news[t] = new
        opts[t] = option
        acs[t] = ac
        prevacs[t] = prevac
        activated_options[t] = active_options_t

        ob, rew, new, _ = env.step(ac)
        rews[t] = rew
        realrews[t] = rew

        curr_opt_duration += 1
        term = pi.get_term([ob], [option])[0][0]

        candidate_option, active_options_t = pi.get_option(ob)
        if term:
            if num_options > 1:
                rews[t] -= dc
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            option = candidate_option
            term_prob = pi.get_tpred([ob], [option])[0][0]

        option_terms.append(term_prob)

        ep_states[option].append(ob)

        cur_ep_ret += rew
        cur_ep_len += 1

        dist = ob[:2] - GOAL
        if np.linalg.norm(dist) < 0.025 and new_rollout:
            insertion = insertion + 1
            new_rollout = False

        if new or (t > 0 and t % horizon == 0):
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
            new_rollout = True

            term_prob = pi.get_tpred([ob], [option])[0][0]

        t += 1

    vpreds, op_vpreds, vpred, op_vpred, op_probs, intfc = pi.get_allvpreds(obs, ob)
    term_ps, term_p = pi.get_alltpreds(obs, ob)
    last_betas = term_ps[range(len(last_options)), last_options]
    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg =  {"ob": obs, "rew": rews, "realrew": realrews, "vpred": vpreds, "op_vpred": op_vpreds, "new": news,
           "ac": acs, "opts": opts, "prevac": prevacs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new),
           "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': term_ps, 'next_term_p': term_p,
           "opt_dur": opt_duration, "op_probs": op_probs, "last_betas": last_betas, "intfc": intfc,
           "activated_options": activated_options, "success": insertion}

    return seg


def add_vtarg_and_adv(seg, gamma, lam, num_options):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    T = len(seg["rew"])
    seg["op_adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * op_vpred[t + 1] * nonterminal - op_vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["op_adv"] = gaelam

    term_p = np.vstack((np.array(seg["term_p"]), np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"], seg["nextvpred"]))
    u_sw = (1 - term_p) * q_sw + term_p * np.tile(op_vpred[:, None], num_options)
    opts = seg["opts"]

    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
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


def learn(env, model_path, policy_func, *,
          batchSize, horizon,
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          retrain=False,
          num_options=1,
          app='',
          saves=False,
          wsaves=False,
          epoch=0,
          seed=1,
          dc=0, plots=False, w_intfc=True, switch=False, intlr=1e-4, piolr=1e-4, fewshot=False, k=0.,
          ):
    optim_batchsize_ideal = optim_batchsize
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy
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

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

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
    lossandgrad_vf = U.function([ob, ac, atarg, ret, lrmult, option], losses + [U.flatgrad(vf_loss, var_list)])
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

    if retrain == True:
        print("Retraining to New Goal")
        time.sleep(2)
        U.load_state(model_path)

    p = []

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

        seg = traj_segment_generator(pi, env, batchSize=batchSize, horizon=horizon, stochastic=True, num_options=2, dc=dc)

        add_vtarg_and_adv(seg, gamma, lam, num_options)

        opt_d = []
        for i in range(num_options):
            dur = np.mean(seg['opt_dur'][i]) if len(seg['opt_dur'][i]) > 0 else 0.
            opt_d.append(dur)
        '''
        print("mean opt dur:", opt_d)
        print("mean op probs:", np.mean(np.array(seg['op_probs']), axis=0))
        print("mean term p:", np.mean(np.array(seg['term_p']), axis=0))
        print("mean vpreds:", np.mean(np.array(seg['vpred']), axis=0))
        '''
        ob, ac, opts, atarg, tdlamret = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy
        assign_old_eq_new()  # set old parameter values to new parameter values

        min_batch = 32  # Arbitrary
        t_advs = [[] for _ in range(num_options)]
        for opt in range(num_options):
            indices = np.where(opts == opt)[0]
            print("batch size:", indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                t_advs[opt].append(0.)
                continue

            ########## This part is only necessary when we use options. We proceed to these verifications in order not to discard any collected trajectories.
            if datas[opt] != 0:
                if (indices.size < min_batch and datas[opt].n > min_batch):
                    datas[opt] = Dataset(
                        dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                        shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif indices.size + datas[opt].n < min_batch:
                    oldmap = datas[opt].data_map

                    cat_ob = np.concatenate((oldmap['ob'], ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'], ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'], atarg[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'], tdlamret[indices]))
                    datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg),
                                         shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (
                        indices.size > min_batch and datas[opt].n < min_batch):

                    oldmap = datas[opt].data_map
                    cat_ob = np.concatenate((oldmap['ob'], ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'], ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'], atarg[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'], tdlamret[indices]))
                    datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg),
                                             shuffle=not pi.recurrent)

                if (indices.size > min_batch and datas[opt].n > min_batch):
                    datas[opt] = d = Dataset(
                        dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                        shuffle=not pi.recurrent)

            elif datas[opt] == 0:
                datas[opt] = d = Dataset(
                    dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                    shuffle=not pi.recurrent)
            #########

            optim_batchsize_corrected = optim_batchsize or ob.shape[0]
            optim_epochs_corrected = np.clip(np.int(indices.size / optim_batchsize_corrected), 1,  optim_epochs)
            print("optim epochs:", optim_epochs)
            logger.log("Optimizing...")

            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs_corrected):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize_corrected):
                    if iters_so_far < 150 or not fewshot:
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                        cur_lrmult, [opt])
                        adam.update(grads, optim_stepsize * cur_lrmult)
                        losses.append(newlosses)
                    else:
                        *newlosses, grads = lossandgrad_vf(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                           cur_lrmult, [opt])
                        adam.update(grads, optim_stepsize * cur_lrmult)
                        losses.append(newlosses)

        if iters_so_far < 150 or not fewshot:
            termg = termgrad(seg["ob"], seg['opts'], seg["op_adv"])[0]
            adam.update(termg, 5e-7)

            if w_intfc:
                intgrads = intgrad(seg['ob'], seg['opts'], seg["last_betas"], seg["op_adv"], seg["op_probs"], seg["activated_options"])[0]
                adam.update(intgrads, intlr)

        opgrads = opgrad(seg['ob'], seg['opts'], seg["last_betas"], seg["op_adv"], seg["intfc"], seg["activated_options"])[0]
        adam.update(opgrads, piolr)
        data = {'seg': seg}
        p.append(data)
        pickle.dump(p, open("data/ioc_data_exp_7b.pkl", "wb"))
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


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
