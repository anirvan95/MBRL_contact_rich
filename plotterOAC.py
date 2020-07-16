import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from model_learning import partialHybridModel
import networkx as nx

# supressing warnings
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

stime = time.time()
f = open("results/MOAC/exp_5/data/rollout_data.pkl", "rb")
p = pickle.load(f)
f.close()

iteration = 50
horizon = 150
rollouts = 20

for iter in range(0, len(p), 5):
    print("Iteration : ", iter)
    data = p[iter]
    seg = data['seg']

    obs = seg['ob']
    acs = seg['ac']
    vpred = seg['vpred']
    opts = seg['opts']
    term = seg['term_p']
    activ_opt = seg['activated_options']
    # Prepare for plotting
    fig = plt.figure()

    for rollout in range(0, rollouts):
        # print("Rollout: ", rollout)
        states = obs[rollout * horizon:(horizon + rollout * horizon), :]
        action = acs[rollout * horizon:(horizon + rollout * horizon), :]
        value = vpred[rollout * horizon:(horizon + rollout * horizon), :]
        time_vect = np.arange(0, len(states)) * 0.01
        options = opts[rollout * horizon:(horizon + rollout * horizon)]
        interest = activ_opt[rollout * horizon:(horizon + rollout * horizon)]
        termination = term[rollout * horizon:(horizon + rollout * horizon), :]
        for t in range(0, len(time_vect)):
            # Trajectory X
            ax = fig.add_subplot(241)
            if options[t] == 0:
                ax.plot(t, states[t, 0], '.r-', label='op_0', linewidth=0.3)
            elif options[t] == 1:
                ax.plot(t, states[t, 0], '.b-', label='op_1', linewidth=0.5)
            elif options[t] == 2:
                ax.plot(t, states[t, 0], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
            elif options[t] == 3:
                ax.plot(t, states[t, 0], '.y-', label='op_3', linewidth=0.7, alpha=0.4)

            # Trajectory Y
            ax = fig.add_subplot(242)
            if options[t] == 0:
                ax.plot(t, states[t, 1], '.r-', label='op_0', linewidth=0.3)
            elif options[t] == 1:
                ax.plot(t, states[t, 1], '.b-', label='op_1', linewidth=0.5)
            elif options[t] == 2:
                ax.plot(t, states[t, 1], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
            elif options[t] == 3:
                ax.plot(t, states[t, 1], '.y-', label='op_3', linewidth=0.7, alpha=0.4)

            # Option Value
            ax = fig.add_subplot(243)
            ax.plot(t, value[t, 0], '.r-', label='op_0', linewidth=0.3)
            ax.plot(t, value[t, 1], '.b-', label='op_1', linewidth=0.5)
            ax.plot(t, value[t, 2], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
            ax.plot(t, value[t, 3], '.y-', label='op_3', linewidth=0.7, alpha=0.4)

            #Actions
            ax = fig.add_subplot(244)
            ax.plot(t, action[t, 0], '.r-', label='op_0', linewidth=0.3)
            ax.plot(t, action[t, 1], '.b-', label='op_1', linewidth=0.5)

            #Interest and term opt 1
            ax = fig.add_subplot(245)
            ax.plot(t, interest[t, 0], '.r-', label='op_0', linewidth=0.3)
            ax.plot(t, termination[t, 0], '.k-', label='op_0', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(246)
            ax.plot(t, interest[t, 1], '.b-', label='op_1', linewidth=0.3)
            ax.plot(t, termination[t, 1], '.k-', label='op_1', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(247)
            ax.plot(t, interest[t, 2], '.g-', label='op_2', linewidth=0.3)
            ax.plot(t, termination[t, 2], '.k-', label='op_2', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(248)
            ax.plot(t, interest[t, 3], '.y-', label='op_3', linewidth=0.3)
            ax.plot(t, termination[t, 3], '.k-', label='op_3', linewidth=0.5, alpha=0.2)

    file_name = 'figures/MOAC/exp_5/iteration_' + str(iter)
    plt.savefig(file_name)
    plt.close()


model = pickle.load(open('results/MOAC/exp_5/model/hybrid_model.pkl','rb'))

print("Model graph:", model.transitionGraph.nodes)
print("Model options:", model.transitionGraph.edges)
edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])

print("Options: ", model.nOptions)
G = model.transitionGraph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, with_labels=True, font_weight='bold', edge_labels=labels)
plt.savefig('figures/MOAC/exp_5/transition_graph.png')

print("Time taken:", time.time() - stime)
