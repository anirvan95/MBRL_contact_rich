import tensorflow as tf
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from common.model_learning_utils import *
from collections import deque


class partialHybridModel(object):
    def __init__(self, params):
        self.transitionGraph = nx.DiGraph()
        self.transitionGraph.add_node('null')
        self.guardFunction = SVMPrediction(params)
        self.modeFunction = SVMPrediction(params)
        self.intFunction = np.zeros((self.preOptions, 1))
        self.termFunction = np.ones((self.preOptions, 1))
        self.model_learning_params = params
        self.modeGaussians = []
        self.nModes = 0
        self.nOptions = -1
        self.horizon = params(1)
        self.dataset = []
        self.preOptions = 10
        self.currentMode = 0
        for m in range(0, params['expectedModes']):
            self.dataset.append(deque(maxlen=100))

    def updateModel(self, rollouts):
        self.learnModes(rollouts)
        self.learnTranstionRelation(rollouts)
        self.learnModeF()
        self.learnGuardF()

    def learnModes(self, rollouts):
        '''
        Performs clustering and assigns modes for possible segments of data
        :param rollouts: rollouts data from environment
        '''
        train_rollouts = int(self.model_learning_params['per_train'])
        obs = rollouts['ob']
        acs = rollouts['ac']
        cFs = rollouts['contactF']
        nFs = rollouts['normalF']
        state_dim = obs.shape(1)
        segmented_traj = []
        segment_dynamics = []
        for rollout in range(0, train_rollouts):
            position = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), 0:int(state_dim/2)]
            # velocity = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), int(state_dim/2):]
            # action = acs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            contactForce = cFs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            normalForce = nFs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            t = np.expand_dims(np.arange(0, len(position)), axis=1)
            feature_vect = np.hstack((t, position, contactForce, normalForce))
            tp = identifyTransitions(feature_vect, self.model_learning_params['window_size'], self.model_learning_params['weight_prior'], self.model_learning_params['n_components'])
            fittedModel, segTraj = fitGaussianDistribution(position, contactForce, tp, self.model_learning_params['minLength'], self.model_learning_params['guassianEps'])
            segmented_traj.append(np.array([rollout, segTraj]))
            if rollout == 0:
                segment_dynamics = fittedModel
            else:
                segment_dynamics = np.concatenate((segment_dynamics, fittedModel), axis=0)
            rollout += 1

        segmented_traj = np.array(segmented_traj)
        db = DBSCAN(eps=self.model_learning_params['DBeps'], min_samples=self.model_learning_params['DBmin_samples'], metric=computeDistance)
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
                    labels[index[0]] = self.nmodes
                    self.modeGaussians.append(mergedClusterGaussian)
                    self.nModes += 1
                else:
                    # Previous Mode
                    closestMode = np.where(distance_vect == np.amin(distance_vect))[0][0]
                    labels[index[0]] = closestMode
                    self.modeGaussians[closestMode] = mergeGaussians(mergedClusterGaussian, self.modeGaussians[closestMode], self.model_learning_params['weightCurrent'], self.model_learning_params['weightPrevious'])

        self.labels = labels
        self.segment_data = segmented_traj

    def learnTranstionRelation(self, rollouts):
        '''
        Creates transtion relations and datasets for model learning
        :param rollouts: rollouts data from environment
        '''
        # TODO: Better approach to avoid last segment
        train_rollouts = int(self.model_learning_params['per_train'])-1
        obs = rollouts['ob']
        acs = rollouts['ac']
        segCount = 0
        for rollout in range(0, train_rollouts):
            states = obs[rollout * self.horizon:(self.horizon + rollout * self.horizon), :]
            numSegments = self.segment_data[rollout][1].shape[0]
            for segment in range(0, numSegments):
                # Ignore Noisy segments
                if self.labels[segCount] >= 0:
                    segStates = states[self.segment_data[rollout][1][segment][0]:(self.segment_data[rollout][1][segment][1] + 1), :]
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
                            self.transitionGraph.add_weighted_edges_from([(self.labels[segCount], 'null', self.nOptions)])

                        # Transition detected
                        if self.labels[segCount] != self.labels[segCount + 1]:
                            if self.transitionGraph.has_edge(self.labels[segCount], 'null'):
                                self.transitionGraph.add_weighted_edges_from([(self.labels[segCount], self.labels[segCount + 1], self.transitionGraph[self.labels[segCount]]['null']['weight'])])
                                self.transitionGraph.remove_edge(self.labels[segCount], 'null')
                            elif not self.transitionGraph.has_edge(self.labels[segCount], self.labels[segCount + 1]):
                                self.nOptions += 1
                                self.transitionGraph.add_weighted_edges_from([(self.labels[segCount], self.labels[segCount + 1], self.nOptions)])

                    if len(list(self.transitionGraph.successors(self.labels[segCount]))) == 0:
                        self.nOptions += 1
                        self.transitionGraph.add_weighted_edges_from([(self.labels[segCount], 'null', self.nOptions)])

    def learnGuardF(self):
        x = []
        y = []
        for mode in range(0, len(self.dataset)):
            modeData = self.dataset[mode]
            for i in range(0, len(modeData)):
                x.append(modeData[i]['x'])
                y.append(modeData[i]['label_t'])
        X = np.array(x)
        Y = np.array(y)
        # Updates the Guard Function
        self.guardFunction.train(X, Y)

    def learnModeF(self):
        x = []
        y = []
        for mode in range(0, len(self.dataset)):
            modeData = self.dataset[mode]
            for i in range(0, len(modeData)):
                x.append(modeData[i]['x'])
                y.append(modeData[i]['label'])
        X = np.array(x)
        Y = np.array(y)
        # Updates the ModeFunction
        self.modeFunction.train(X, Y)

    def getInterest(self, ob):
        predMode = self.modeFunction.predict(ob)
        mode = self.currentMode
        self.intFunction = np.zeros((self.preOptions, 1))
        nextModes = self.transitionGraph.successors(mode)
        for nextMode in range(0, nextModes):
            option = self.transitionGraph[mode][nextMode]['weight']
            self.intFunction[option] = 1

    def getInterestAdv(self, ob):
        mode = self.currentMode
        modes_prob = self.modeFunction.predict_f(ob)
        nextModes = self.transitionGraph.successors(mode)
        norm_modes_prob = modes_prob[nextModes]/sum(modes_prob[nextModes])
        self.intFunction = np.zeros((self.preOptions, 1))
        for nextMode in range(0, nextModes):
            optionInd = self.transitionGraph[mode][nextMode]['weight']
            self.intFunction[optionInd] = norm_modes_prob[optionInd]

    def getTermination(self, ob):
        # TODO: Check here for termination query
        mode = self.currentMode
        next_modes_prob = self.guardFunction.predict_f(ob)
        # Normalize
        neighbourModes = self.transitionGraph.successors(mode) + self.transitionGraph.predecessors(mode)
        norm_nmode_prob = next_modes_prob[neighbourModes]/sum(next_modes_prob[neighbourModes])
        term_prob = 1 - norm_nmode_prob
        self.termFunction = np.ones((self.preOptions, 1))
        nextModes = self.transitionGraph.successors(mode)
        for nextMode in range(0, nextModes):
            optionInd = self.transitionGraph[mode][nextMode]['weight']
            self.termFunction[optionInd] = term_prob

    def getNextMode(self, ob):
        mode = self.modeFunction.predict(ob)
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

    print("Mode Dynamics learned\n")
