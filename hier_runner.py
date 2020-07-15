import tensorflow.compat.v1 as tf1
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
from model_learning import partialHybridModel

tf1.disable_v2_behavior()


# collect trajectory
def sample_trajectory(pi, model, env, horizon=150, rollouts=50, render=False):
    """
            Generates rollouts for policy optimization
    """
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
    batch_size = int(horizon * rollouts)
    # Initialise history of arrays
    obs = np.array([ob for _ in range(batch_size)])
    rews = np.zeros(batch_size, 'float32')
    news = np.zeros(batch_size, 'int32')
    opts = np.zeros(batch_size, 'int32')
    activated_options = np.zeros((batch_size, num_options), 'float32')

    last_options = np.zeros(batch_size, 'int32')
    acs = np.array([ac for _ in range(batch_size)])
    prev_acs = acs.copy()
    model.currentMode = 0
    option, active_options_t = pi.get_option(ob)
    last_option = option

    betas = []
    vpreds = []
    op_vpreds = []

    opt_duration = [[] for _ in range(num_options)]
    sample_index = 0
    curr_opt_duration = 0

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
        prev_acs[sample_index] = prevac
        beta, vpred, op_vpred = pi.get_preds(ob)

        betas.append(beta)
        vpreds.append(vpred * (1 - new))
        op_vpreds.append(op_vpred)
        activated_options[sample_index] = active_options_t

        # Step in the environment
        ob, rew, new, _ = env.step(ac)

        rews[sample_index] = rew
        curr_opt_duration += 1
        # check if current option is about to end in this state
        nbeta = pi.get_tpred(ob)
        tprob = nbeta[option]

        if render:
            env.render()
        # termination =
        if tprob >= pi.term_prob:
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            last_option = option
            model.currentMode = model.getNextMode(ob)
            option, active_options_t = pi.get_option(ob)

        cur_ep_ret += rew
        cur_ep_len += 1
        dist = env.getGoalDist()

        if np.linalg.norm(dist) < 0.025 and not successFlag:
            success = success + 1
            successFlag = True

        sample_index += 1

        if new or (sample_index > 0 and sample_index % horizon == 0):
            render = False
            env.setRender(False)
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            option, active_options_t = pi.get_option(ob)
            last_option = option
            successFlag = False
            new = True

    env.close()
    betas = np.array(betas)
    vpreds = np.array(vpreds).reshape(batch_size, num_options)
    op_vpreds = np.squeeze(np.array(op_vpreds))
    last_betas = betas[range(len(last_options)), last_options]
    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg = {"ob": obs, "rew": rews, "vpred": np.array(vpreds), "op_vpred": np.array(op_vpreds), "new": news,
           "ac": acs, "opts": opts, "prevac": prev_acs, "nextvpred": vpred * (1 - new),
           "nextop_vpred": op_vpred * (1 - new), "ep_rets": ep_rets, "ep_lens": ep_lens, 'term_p': betas,
           'next_term_p': beta, "last_betas": last_betas,
           "opt_dur": opt_duration, "activated_options": activated_options, "success": success}

    return seg


def add_vtarg_and_adv(seg, gamma, lam, num_options):
    """
        Compute advantage and other value functions using GAE
    """
    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    term_p = np.vstack((np.array(seg["term_p"]), np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"], seg["nextvpred"]))
    # Utility function in option framework
    u_sw = (1 - term_p) * q_sw + term_p * np.tile(op_vpred[:, None], num_options)

    opts = seg["opts"]
    new = np.append(seg["new"], True)
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


def learn(env, model_path, data_path, policy_fn, model_learning_params, svm_grid_params, svm_params_interest,
          svm_params_guard, *, modes, rollouts, num_options=2,
          horizon,  # timesteps per actor per update
          clip_param, ent_coeff=0.02,  # clipping parameter epsilon, entropy coeff
          optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=160,  # optimization hypers
          gamma=0.99, lam=0.95,  # advantage estimation
          max_iters=0,  # time constraint
          adam_epsilon=1.2e-4,
          schedule='linear',  # annealing for stepsize parameters (epsilon and adam)
          retrain=False
          ):
    """
            Core learning function
    """

    ob_space = env.observation_space
    ac_space = env.action_space

    model = partialHybridModel(env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard,
                               horizon, modes, num_options, rollouts)
    pi = policy_fn("pi", ob_space, ac_space, model, num_options)  # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space, model, num_options)  # Network for old policy
    atarg = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Empirical return

    lrmult = tf1.placeholder(name='lrmult', dtype=tf1.float32,
                             shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    # Define placeholders for computing the advantage
    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    betas = tf1.placeholder(dtype=tf1.float32, shape=[None])  # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    # Defining losses for optimization
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf1.reduce_mean(kloldnew)
    meanent = tf1.reduce_mean(ent)
    pol_entpen = (-ent_coeff) * meanent

    ratio = tf1.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf1.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf1.reduce_mean(tf1.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = tf1.reduce_mean(tf1.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf1.assign(oldv, newv) for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=5)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=5)  # rolling buffer for episode rewards

    p = []  # for saving the rollouts

    if retrain == True:
        print("Retraining the model")
        time.sleep(2)
        U.load_state(model_path)
        #model = pickle.load()
    max_timesteps = int(horizon * rollouts * max_iters)

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
        print("Collecting samples for policy optimization !! ")
        render = False
        seg = sample_trajectory(pi, model, env, horizon=horizon, rollouts=rollouts, render=render)
        data = {'seg': seg}
        p.append(data)
        del data
        data_file_name = data_path + '/rollout_data.pkl'
        pickle.dump(p, open(data_file_name, "wb"))

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

        # Optimizing the policy

        for opt in range(num_options):
            indices = np.where(opts == opt)[0]
            print("Option: ", opt)
            print("Batch Size:", indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                continue

            datas[opt] = d = Dataset(
                dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]),
                shuffle=not pi.recurrent)

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
                    adam.update(grads, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)

        # Model update
        print("Updating model ")
        model.updateModel(seg)
        print("Model graph:", model.transitionGraph.nodes)
        print("Model options:", model.transitionGraph.edges)
        edges = list(model.transitionGraph.edges)
        for i in range(0, len(edges)):
            print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])
        # assert(model.nOptions < pi.num_options)

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("Success", seg["success"])
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

    return pi, model
