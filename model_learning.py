import networkx as nx
from sklearn.cluster import DBSCAN
from common.model_learning_utils import *
from collections import deque
from sklearn.exceptions import NotFittedError


class partialHybridModel(object):
    def __init__(self, env, model_learning_params, svm_grid_params, svm_params_interest, svm_params_guard, horizon, modes,
                 options, rollout):
        # Define the transition graph with single mode
        self.transitionGraph = nx.DiGraph()
        self.transitionGraph.add_node('null')
        self.prevModes = len(list(self.transitionGraph.nodes))
        self.nOptions = 0
        self.transitionGraph.add_weighted_edges_from([(0, 'null', self.nOptions)])
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

    def updateModel(self, rollouts):
        # self.learnModes(rollouts)
        # self.learnTranstionRelation(rollouts)
        self.learnHardTG(rollouts)
        self.learnModeF()
        self.learnGuardF()

    def learnHardTG(self, rollouts):
        train_rollouts = int(self.model_learning_params['per_train'] * self.rolloutSize)
        obs = rollouts['ob']
        self.transitionUpdated = False
        for rollout in range(0, train_rollouts):
            states = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            for t in range(0, len(states) - 1):
                label = obtainMode(self.env_id, states[t, :])
                label_t = obtainMode(self.env_id, states[t + 1, :])
                dataDict = {'x': states[t, :], 'label': label, 'label_t': label_t}
                self.dataset[int(label)].append(dataDict)

                # Creating transition graph
                if not self.transitionGraph.has_node(label):
                    self.nOptions += 1
                    self.transitionGraph.add_weighted_edges_from([(label, 'null', self.nOptions)])

                if label != label_t:
                    # Transition Detected
                    if not (self.transitionGraph.has_edge(label, label_t) or self.transitionGraph.has_edge(label_t, label)):
                        if self.transitionGraph.has_edge(label, 'null'):
                            self.transitionGraph.add_weighted_edges_from([(label, label_t, self.transitionGraph[label]['null']['weight'])])
                            self.transitionGraph.remove_edge(label, 'null')
                        else:
                            self.nOptions += 1
                            self.transitionGraph.add_weighted_edges_from([(label, label_t, self.nOptions)])

        nodes = list(self.transitionGraph.nodes)
        for i in range(0, len(nodes)):
            if len(list(self.transitionGraph.successors(nodes[i]))) < 1 and nodes[i] != 'null':
                self.nOptions += 1
                self.transitionGraph.add_weighted_edges_from([(nodes[i], 'null', self.nOptions)])

        if len(list(self.transitionGraph.nodes)) > self.prevModes:
            self.prevModes = len(list(self.transitionGraph.nodes))
            self.transitionUpdated = True

    def learnModes(self, rollouts):
        '''
        Performs clustering and assigns modes for possible segments of data
        :param rollouts: rollouts data from environment
        '''
        train_rollouts = int(self.model_learning_params['per_train'])
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
                                     self.model_learning_params['weight_prior'], self.model_learning_params['n_components'])
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
        db = DBSCAN(eps=self.clustering_params['DBeps'], min_samples=self.clustering_params['DBmin_samples'],
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

                if np.amin(distance_vect) > self.clustering_params['gaussDist']:
                    # New mode
                    labels[index[0]] = self.nmodes
                    self.modeGaussians.append(mergedClusterGaussian)
                    self.nModes += 1
                else:
                    # Previous Mode
                    closestMode = np.where(distance_vect == np.amin(distance_vect))[0][0]
                    labels[index[0]] = closestMode
                    self.modeGaussians[closestMode] = mergeGaussians(mergedClusterGaussian,
                                                                     self.modeGaussians[closestMode],
                                                                     self.clustering_params['weightCurrent'],
                                                                     self.clustering_params['weightPrevious'])

        self.labels = labels
        self.segment_data = segmented_traj

    def learnTranstionRelation(self, rollouts):
        '''
        Creates transtion relations and datasets for model learning
        :param rollouts: rollouts data from environment
        '''
        # TODO: Better approach to avoid last segment
        train_rollouts = int(self.clustering_params['per_train']) - 1
        obs = rollouts['ob']
        acs = rollouts['ac']
        segCount = 0
        for rollout in range(0, train_rollouts):
            states = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            numSegments = self.segment_data[rollout][1].shape[0]
            for segment in range(0, numSegments):
                # Ignore Noisy segments
                if self.labels[segCount] >= 0:
                    segStates = states[self.segment_data[rollout][1][segment][0]:(
                                self.segment_data[rollout][1][segment][1] + 1), :]
                    segLen = len(segStates)
                    segModes = self.labels[segCount] * np.ones((segLen, 1))
                    for t in range(0, segLen):
                        if t < segLen - 1:
                            dataDict = {'x': segStates[t], 'mode': segModes[t], 'mode_t': segModes[t + 1]}
                            self.dataset[int(segModes[t])].append(dataDict)
                        else:  # account for the last state in the segment
                            dataDict = {'x': segStates[t], 'mode': segModes[t], 'mode_t': segModes[segCount + 1]}
                            self.dataset[int(segModes[t])].append(dataDict)

                    # Creating transition graph
                    if self.labels[segCount + 1] >= 0 and segment < numSegments - 1:
                        if not self.transitionGraph.has_node(self.labels[segCount]):
                            self.nOptions += 1
                            self.transitionGraph.add_weighted_edges_from(
                                [(self.labels[segCount], 'null', self.nOptions)])

                        # Transition detected
                        if self.labels[segCount] != self.labels[segCount + 1]:
                            if self.transitionGraph.has_edge(self.labels[segCount], 'null'):
                                self.transitionGraph.add_weighted_edges_from([(self.labels[segCount],
                                                                               self.labels[segCount + 1],
                                                                               self.transitionGraph[
                                                                                   self.labels[segCount]]['null'][
                                                                                   'weight'])])
                                self.transitionGraph.remove_edge(self.labels[segCount], 'null')
                            elif not self.transitionGraph.has_edge(self.labels[segCount], self.labels[segCount + 1]):
                                self.nOptions += 1
                                self.transitionGraph.add_weighted_edges_from(
                                    [(self.labels[segCount], self.labels[segCount + 1], self.nOptions)])

                    if len(list(self.transitionGraph.successors(self.labels[segCount]))) == 0:
                        self.nOptions += 1
                        self.transitionGraph.add_weighted_edges_from([(self.labels[segCount], 'null', self.nOptions)])

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
        # predMode = self.modeFunction.predict([ob])
        mode = self.currentMode
        self.intFunction = np.zeros(self.preOptions)
        nextModes = list(self.transitionGraph.successors(mode))
        for i in range(0, len(nextModes)):
            option = self.transitionGraph[mode][nextModes[i]]['weight']
            self.intFunction[option] = 1

        return self.intFunction

    def getInterestAdv(self, ob):
        # TODO:Under development
        mode = self.currentMode
        modes_prob = self.modeFunction.predict_f(ob)
        nextModes = self.transitionGraph.successors(mode)
        norm_modes_prob = modes_prob[nextModes] / sum(modes_prob[nextModes])
        self.intFunction = np.zeros(self.preOption)
        for nextMode in range(0, nextModes):
            optionInd = self.transitionGraph[mode][nextMode]['weight']
            self.intFunction[optionInd] = norm_modes_prob[optionInd]

        return self.intFunction

    def getTermination(self, ob):
        mode = self.currentMode
        self.termFunction = np.ones(self.preOptions)
        local_group = list(self.transitionGraph.successors(mode)) + list(
            self.transitionGraph.predecessors(mode)) + list([mode])
        if 'null' in local_group:
            local_group.remove('null')
        local_group.sort()
        complete_graph = list(self.transitionGraph.nodes)
        complete_graph.remove('null')
        complete_graph.sort()

        if len(list(self.transitionGraph.nodes)) > 2:
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
                optionInd = self.transitionGraph[mode][nextModes[i]]['weight']
                self.termFunction[optionInd] = term_prob
        else:
            # Only one mode discovered, should not have any termination
            optionInd = self.transitionGraph[mode]['null']['weight']
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
