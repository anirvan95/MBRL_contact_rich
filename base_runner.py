from common.dataset import Dataset
import logger
import common.tf_util as U
import tensorflow.compat.v1 as tf
import numpy as np
import time
from common.mpi_adam import MpiAdam
from common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from common.console_util import fmt_row
from common.math_util import explained_variance, zipsame, flatten_lists
import pickle


def sample_trajectory(pi, env, horizon=150, batch_size=12000, stochastic=True, render=False):
    GOAL = np.array([0.05, 0.05, 0.05])
    sampleInd = 0
    if render:
        env.setRender(True)
    else:
        env.setRender(False)
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    cF = env.getContactForce()
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(batch_size)])
    cFs = np.array([cF for _ in range(batch_size)])
    rews = np.zeros(batch_size, 'float32')
    vpreds = np.zeros(batch_size, 'float32')
    news = np.zeros(batch_size, 'int32')
    acs = np.array([ac for _ in range(batch_size)])
    prevacs = acs.copy()

    insertion = 0
    new_rollout = True
    t = 0
    while sampleInd < batch_size:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        #Store state, action pair
        obs[sampleInd] = ob
        acs[sampleInd] = ac
        prevacs[sampleInd] = prevac
        vpreds[sampleInd] = vpred * (1 - new)
        news[sampleInd] = new

        #print(observations[t] - obs[sampleInd])
        # Take step in environment
        ob, rew, new, _ = env.step(ac)
        if render:
            env.render()
        rews[sampleInd] = rew
        cFs[sampleInd] = env.getContactForce()
        #print(cFs[sampleInd])
        cur_ep_ret += rew
        cur_ep_len += 1

        dist = ob[:3]-GOAL
        if np.linalg.norm(dist) < 0.025 and new_rollout:
            insertion = insertion+1
            new_rollout = False

        t += 1
        sampleInd += 1

        if new or (t > 0 and t == horizon):
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            new = True #Done is true
            env.close()
            env.setRender(False)
            ob = env.reset()
            new_rollout = True
            render = False
            t = 0

    env.close()
    print("\n Maximum Reward this iteration: ", max(ep_rets), " \n")
    seg = {"ob": obs, "rew": rews, "vpred": vpreds, "new": news, "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new), "ep_rets": ep_rets, "ep_lens": ep_lens, "success": insertion, 'contactF': cFs}
    return seg


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 1) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, model_path, policy_fn, *,
        horizon=150, # timesteps per actor per update
        batch_size_per_episode=15000,
        clip_param=0.2, entcoeff=0.02, # clipping parameter epsilon, entropy coeff
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,# optimization hypers
        gamma=0.99, lam=0.95, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-4,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        retrain=False
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=5) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=5) # rolling buffer for episode rewards

    p = [] #for saving the rollouts

    if retrain == True:
        print("Retraining to New Goal")
        time.sleep(2)
        U.load_state(model_path)

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
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        if iters_so_far >= 90:
            render = True
        else:
            render = False
        seg = sample_trajectory(pi, env, horizon=horizon, batch_size=batch_size_per_episode, stochastic=True, render=render)
        data = {'seg': seg}
        p.append(data)
        del data
        pickle.dump(p, open("data/base_actor_critic_blockSlideExp2.pkl", "wb"))

        add_vtarg_and_adv(seg, gamma, lam)

        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), deterministic=pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
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
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    return pi

