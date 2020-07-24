import networkx as nx
from sklearn.cluster import DBSCAN
from common.model_learning_utils import *
from collections import deque
from sklearn.exceptions import NotFittedError
import xlwt
from xlwt import Workbook


class partialHybridModel(object):
    def __init__(self, env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes, options, rollout):
        # Define the transition graph with single mode
        self.transitionGraph = nx.DiGraph()
        self.transitionGraph.add_node('goal')
        self.prevModes = len(list(self.transitionGraph.nodes))
        self.nOptions = 0
        self.transitionGraph.add_weighted_edges_from([(0, 'goal', self.nOptions)])
        self.env_id = env.unwrapped.spec.id
        self.preOptions = options
        self.modeGaussians = []
        self.nModes = 0
        self.horizon = horizon
        self.dataset = []
        self.currentMode = 0
        self.transitionUpdated = False
        self.rolloutSize = rollout
        self.guardFunction = SVMPrediction(svm_params_interest, svm_grid_params)
        self.modeFunction = SVMPrediction(svm_params_guard, svm_grid_params)
        self.intFunction = np.zeros(self.preOptions)
        self.termFunction = np.ones(self.preOptions)
        self.model_learning_params = model_learning_params
        for m in range(0, modes):
            self.dataset.append(deque(maxlen=model_learning_params['queueSize']))

    def updateModel(self, rollouts, pi):
        # self.learnModes(rollouts)
        self.learnPreDefModes(rollouts)
        self.learnTranstionRelation(rollouts, pi)
        self.learnGuardF()
        self.learnModeF()

    def learnPreDefModes(self, rollouts):
        '''
        Assigns modes and segments the rollouts according to pre-defined regions
        :param rollouts: rollouts data from environment
        '''
        train_rollouts = int(self.model_learning_params['per_train']) * self.rolloutSize
        obs = rollouts['ob']
        segmented_traj = []
        seg_labels = []
        for rollout in range(0, train_rollouts):
            states = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            traj_time = len(states)
            tp = []
            for t in range(0, traj_time - 1):
                label = obtainMode(self.env_id, states[t, :])
                label_t = obtainMode(self.env_id, states[t + 1, :])
                dataDict = {'x': states[t, :], 'label': label, 'label_t': label_t}
                self.dataset[int(label)].append(dataDict)
                if label != label_t:
                    tp.append(t)
            tp.append(0)
            tp.append(traj_time)
            tp.sort()
            tp = np.array(tp)
            selectedSeg = []
            for k in range(0, len(tp) - 1):
                if (tp[k + 1] - tp[k]) > self.model_learning_params['minLength']:
                    selectedSeg.append(np.array([tp[k], tp[k + 1]]))
                    seg_labels.append(obtainSegMode(self.env_id, states[tp[k]:tp[k + 1], :]))
            segmented_traj.append(np.array([rollout, selectedSeg]))

        self.labels = np.array(seg_labels)
        self.segment_data = np.array(segmented_traj)

    def learnModes(self, rollouts):
        '''
        Performs clustering and assigns modes for possible segments of data
        :param rollouts: rollouts data from environment
        '''
        train_rollouts = int(self.model_learning_params['per_train']) * self.rolloutSize
        obs = rollouts['ob']
        acs = rollouts['ac']
        cFs = rollouts['contactF']
        state_dim = obs.shape(1)
        segmented_traj = []
        segment_dynamics = []
        for rollout in range(0, train_rollouts):
            position = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), 0:int(state_dim / 2)]
            # velocity = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), int(state_dim/2):]
            # action = acs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            contactForce = cFs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            t = np.expand_dims(np.arange(0, len(position)), axis=1)
            feature_vect = np.hstack((t, position, contactForce))
            tp = identifyTransitions(feature_vect, self.model_learning_params['window_size'],
                                     self.model_learning_params['weight_prior'],
                                     self.model_learning_params['n_components'])
            fittedModel, segTraj = fitGaussianDistribution(position, contactForce, tp,
                                                           self.model_learning_params['minLength'],
                                                           self.model_learning_params['guassianEps'])
            segmented_traj.append(np.array([rollout, segTraj]))
            if rollout == 0:
                segment_dynamics = fittedModel
            else:
                segment_dynamics = np.concatenate((segment_dynamics, fittedModel), axis=0)
            rollout += 1

        segmented_traj = np.array(segmented_traj)
        db = DBSCAN(eps=self.model_learning_params['DBeps'], min_samples=self.model_learning_params['DBmin_samples'],
                    metric=computeDistance)
        labels = db.fit_predict(segment_dynamics)
        nClusters = len(set(labels)) - (1 if -1 in labels else 0)
        nNoise = list(labels).count(-1)
        # print('Estimated number of clusters: %d' % nClusters)
        # print('Estimated number of noise points: %d' % nNoise)

        # Correcting Labels -> Modes
        for cluster in range(0, nClusters):
            index = np.where(labels == cluster)
            clusterGaussian = segment_dynamics[index[0], :]
            mergedClusterGaussian = clusterGaussian[0, :]
            for i in range(1, len(clusterGaussian)):
                mergedClusterGaussian = mergeGaussians(mergedClusterGaussian, clusterGaussian[i, :], 0.5, 0.5)

            if len(self.modeGaussians) == 0:
                self.nModes += 1
                self.modeGaussians.append(clusterGaussian)
            else:
                distance_vect = np.zeros(len(self.modeGaussians))
                for i in range(0, len(self.modeGaussians)):
                    distance_vect[i] = computeDistance(self.modeGaussians[i], mergedClusterGaussian)

                if np.amin(distance_vect) > self.model_learning_params['gaussDist']:
                    # New mode
                    labels[index[0]] = self.nModes
                    self.modeGaussians.append(mergedClusterGaussian)
                    self.nModes += 1
                else:
                    # Previous Mode
                    closestMode = np.where(distance_vect == np.amin(distance_vect))[0][0]
                    labels[index[0]] = closestMode
                    self.modeGaussians[closestMode] = mergeGaussians(mergedClusterGaussian,
                                                                     self.modeGaussians[closestMode],
                                                                     self.model_learning_params['weightCurrent'],
                                                                     self.model_learning_params['weightPrevious'])

        self.labels = labels
        self.segment_data = segmented_traj

    def learnTranstionRelation(self, rollouts, pi):
        '''
        Creates transtion relations and desired mode transition options and updates the dataset
        :param rollouts: rollouts data from environment
        :param pi: policy
        '''
        # TODO: Better approach to avoid last segment, merge segments with same modes
        train_rollouts = int(self.model_learning_params['per_train']) * self.rolloutSize
        self.transitionUpdated = False
        obs = rollouts['ob']
        acs = rollouts['ac']
        rews = rollouts['rew']
        news = rollouts['new']
        opts = rollouts['opts']
        des_opts_seg = np.zeros(self.labels.shape)
        des_opts = []
        imp_samp = []
        des_act = []

        seg_obs = []
        seg_acs = []
        seg_opts = []
        seg_rews = []
        seg_news = []

        seg_count = 0
        for rollout in range(0, train_rollouts - 1):
            option = opts[rollout * self.horizon:(self.horizon + rollout * self.horizon)]
            states = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            action = acs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            rewards = rews[rollout * self.horizon:(self.horizon + rollout * self.horizon)]
            episode = news[rollout * self.horizon:(self.horizon + rollout * self.horizon)]

            num_segments = len(self.segment_data[rollout][1])
            for segment in range(0, num_segments):
                # Avoid Noisy segments
                if self.labels[seg_count] >= 0:
                    # Adding mode to GOAL edge e.g. 0 > goal, 1 > goal etc
                    if not self.transitionGraph.has_edge(self.labels[seg_count], 'goal'):
                        self.nOptions += 1
                        self.transitionGraph.add_weighted_edges_from(
                            [(self.labels[seg_count], 'goal', self.nOptions)])
                        self.transitionUpdated = False

                    if self.labels[seg_count + 1] >= 0:  # Avoid noisy next segment
                        # Adding mode to GOAL edge e.g. 1 > goal, 1 > goal etc
                        if not self.transitionGraph.has_edge(self.labels[seg_count + 1], 'goal'):
                            self.nOptions += 1
                            self.transitionGraph.add_weighted_edges_from([(self.labels[seg_count + 1], 'goal', self.nOptions)])
                            self.transitionUpdated = False

                        # Checking for transition detection while ignoring the last segment
                        if self.labels[seg_count] != self.labels[seg_count + 1] and segment < num_segments - 1:
                            if not (self.transitionGraph.has_edge(self.labels[seg_count], self.labels[seg_count + 1])):
                                self.nOptions += 1
                                self.transitionGraph.add_weighted_edges_from([(self.labels[seg_count], self.labels[seg_count + 1], self.nOptions)])

                            # Assign the desired option for transition
                            # 0>1
                            if self.labels[seg_count + 1] > self.labels[seg_count]:
                                des_opts_seg[seg_count] = self.transitionGraph[self.labels[seg_count]][self.labels[seg_count + 1]]['weight']
                            # 1>0 gets assigned to 1>goal
                            else:
                                des_opts_seg[seg_count] = self.transitionGraph[self.labels[seg_count]]['goal']['weight']

                        # Assigning desired option for last segment or only one segment
                        else:
                            des_opts_seg[seg_count] = self.transitionGraph[self.labels[seg_count]]['goal']['weight']

                    # Creating the updated database
                    segStates = states[self.segment_data[rollout][1][segment][0]:self.segment_data[rollout][1][segment][1], :]
                    segAction = action[self.segment_data[rollout][1][segment][0]:self.segment_data[rollout][1][segment][1], :]
                    segOpts = option[self.segment_data[rollout][1][segment][0]:self.segment_data[rollout][1][segment][1]]
                    segReward = rewards[self.segment_data[rollout][1][segment][0]:self.segment_data[rollout][1][segment][1]]
                    segEpisode = episode[self.segment_data[rollout][1][segment][0]:self.segment_data[rollout][1][segment][1]]

                    if seg_count == 0:
                        seg_obs = segStates
                        seg_opts = segOpts
                        seg_acs = segAction
                        seg_rews = segReward
                        seg_news = segEpisode
                    else:
                        seg_obs = np.vstack((seg_obs, segStates))
                        seg_acs = np.vstack((seg_acs, segAction))
                        seg_opts = np.append(seg_opts, segOpts)
                        seg_rews = np.append(seg_rews, segReward)
                        seg_news = np.append(seg_news, segEpisode)
                    # Compute importance sampling ration
                    is_ratio = 1
                    for t in range(0, len(segStates)):
                        desired_mean, desired_std = pi.get_ac_dist(segStates[t, :], des_opts_seg[seg_count])
                        actual_mean, actual_std = pi.get_ac_dist(segStates[t, :], segOpts[t])
                        is_ratio = is_ratio * compute_likelihood(desired_mean, desired_std, segAction[t, :]) / compute_likelihood(actual_mean, actual_std, segAction[t, :])
                        imp_samp.append(is_ratio)
                        des_opts.append(des_opts_seg[seg_count])

                seg_count += 1

        rollouts['is'] = np.array(imp_samp)
        rollouts['des_opts'] = np.array(des_opts)
        rollouts['seg_obs'] = seg_obs
        rollouts['seg_acs'] = seg_acs
        rollouts['seg_opts'] = seg_opts
        rollouts['seg_rews'] = seg_rews
        rollouts['seg_news'] = seg_news

    def learnGuardF(self):

        x = []
        y = []
        if len(list(self.transitionGraph.nodes)) > 2:
            for mode in range(0, len(self.dataset)):
                modeData = self.dataset[mode]
                print(len(modeData))
                for i in range(0, len(modeData)):
                    x.append(modeData[i]['x'])
                    y.append(modeData[i]['label_t'])
            X = np.array(x)
            Y = np.array(y)
            # Updates the Guard Function
            if self.transitionUpdated:
                self.guardFunction.train(X, Y, True)
            else:
                self.guardFunction.train(X, Y, False)

    def learnModeF(self):
        x = []
        y = []
        if len(list(self.transitionGraph.nodes)) > 2:
            for mode in range(0, len(self.dataset)):
                modeData = self.dataset[mode]
                for i in range(0, len(modeData)):
                    x.append(modeData[i]['x'])
                    y.append(modeData[i]['label'])
            X = np.array(x)
            Y = np.array(y)
            # Updates the ModeFunction
            if self.transitionUpdated:
                self.modeFunction.train(X, Y, True)
            else:
                self.modeFunction.train(X, Y, False)

    def getInterest(self, ob):
        mode = self.currentMode
        self.intFunction = np.zeros(self.preOptions)
        if len(list(self.transitionGraph.nodes)) > 2:
            mode = self.modeFunction.predict([ob])[0]
            nextModes = list(self.transitionGraph.successors(mode))
            for i in range(0, len(nextModes)):
                if nextModes[i] == 'goal' or nextModes[i] > mode:
                    option = self.transitionGraph[mode][nextModes[i]]['weight']
                    self.intFunction[option] = 1
        else:
            # Only one mode discovered, should select only that option
            optionInd = self.transitionGraph[mode]['goal']['weight']
            self.intFunction[optionInd] = 1

        return self.intFunction

    def getTermination(self, ob):
        mode = self.currentMode
        self.termFunction = np.ones(self.preOptions)

        if len(list(self.transitionGraph.nodes)) > 2:
            mode = self.modeFunction.predict([ob])[0]
            local_group = list(self.transitionGraph.successors(mode)) + list(self.transitionGraph.predecessors(mode)) + list([mode])
            if 'goal' in local_group:
                local_group.remove('goal')
            local_group.sort()
            # remove duplicate elements
            local_group = np.unique(np.array(local_group))
            complete_graph = list(self.transitionGraph.nodes)
            complete_graph.remove('goal')
            complete_graph.sort()
            next_modes_prob = self.guardFunction.predict_f([ob])[0]
            # Normalize
            sum_prob = 0
            for i in range(0, len(local_group)):
                mode_index = complete_graph.index(local_group[i])
                sum_prob += next_modes_prob[mode_index]
            norm_nmode_prob = next_modes_prob[complete_graph.index(mode)] / sum_prob
            term_prob = 1 - norm_nmode_prob
            nextModes = list(self.transitionGraph.successors(mode))
            for i in range(0, len(nextModes)):
                if nextModes[i] == 'goal' or nextModes[i] > mode:
                    optionInd = self.transitionGraph[mode][nextModes[i]]['weight']
                    self.termFunction[optionInd] = term_prob
        else:
            # Only one mode discovered, should not have any termination
            optionInd = self.transitionGraph[mode]['goal']['weight']
            self.termFunction[optionInd] = 0

        return self.termFunction

    def getNextMode(self, ob):
        try:
            mode = self.modeFunction.predict([ob])[0]
        except NotFittedError as e:
            mode = self.currentMode
        return mode


class HybridModel(partialHybridModel):
    def __init__(self, params):
        self.expert_gp_params = params
        self.returnMap_params = params
        self.returnMap = multiDimGaussianProcess(self.returnMap_params)
        self.modeDynamicsModel = []
        self.expert_gp_params = params

    def learnReturnMaps(self, X, Y):
        self.returnMap.fit(X, Y)

    def learnModeDynamics(self, nmodes, x_train, delx_train, u_train, label_train, label_t_train):

        self.modeDynamicsModel = []
        for mode in range(0, nmodes):
            x_train_mode = x_train[(np.logical_and((label_train == mode), (label_t_train == mode)))]
            delx_train_mode = delx_train[(np.logical_and((label_train == mode), (label_t_train == mode)))]
            u_train_mode = u_train[(np.logical_and((label_train == mode), (label_t_train == mode)))]
            X = np.hstack((x_train_mode, u_train_mode))
            Y = delx_train_mode
            modeDynamics = multiDimGaussianProcess(self.expert_gp_params)
            modeDynamics.fit(X, Y)
            self.modeDynamicsModel.append(modeDynamics)
