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
f = open("results/MOAC/isexp2_7/data/rollout_data.pkl", "rb")
p = pickle.load(f)
f.close()

iteration = 25
horizon = 200
rollouts = 5

for iter in range(0, len(p), 3):
    print("Iteration : ", iter)
    data = p[iter]
    seg = data['rollouts']

    obs = seg['seg_obs']
    acs = seg['seg_acs']
    vpred = seg['vpreds']
    opts = seg['seg_opts']
    des_opts = seg['des_opts']
    term = seg['betas']
    activ_opt = seg['activated_options']
    advs = seg["adv"]
    is_values = seg['is']
    # Prepare for plotting
    fig = plt.figure()

    for rollout in range(0, rollouts):
        # print("Rollout: ", rollout)
        states = obs[rollout * horizon:(horizon + rollout * horizon), :]
        action = acs[rollout * horizon:(horizon + rollout * horizon), :]
        value = vpred[rollout * horizon:(horizon + rollout * horizon), :]
        time_vect = np.arange(0, len(states)) * 0.01
        options = opts[rollout * horizon:(horizon + rollout * horizon)]
        des_options = des_opts[rollout * horizon:(horizon + rollout * horizon)]
        interest = activ_opt[rollout * horizon:(horizon + rollout * horizon)]
        termination = term[rollout * horizon:(horizon + rollout * horizon), :]
        adv = advs[rollout * horizon:(horizon + rollout * horizon)]
        is_val = is_values[rollout * horizon:(horizon + rollout * horizon)]
        for t in range(0, len(time_vect)):
            # Selected options
            # Trajectory X
            ax = fig.add_subplot(3, 4, 1)
            if options[t] == 0:
                ax.plot(t, states[t, 0], '.r-', label='op_0', linewidth=0.3)
            elif options[t] == 2:
                ax.plot(t, states[t, 0], '.b-', label='op_1', linewidth=0.5)
            elif options[t] == 1:
                ax.plot(t, states[t, 0], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
            elif options[t] == 5:
                ax.plot(t, states[t, 0], '.c-', label='op_4', linewidth=0.7, alpha=0.4)
            elif options[t] == 4:
                ax.plot(t, states[t, 0], '.m-', label='op_5', linewidth=0.7, alpha=0.4)


            # Trajectory Y
            ax = fig.add_subplot(3, 4, 2)
            if options[t] == 0:
                ax.plot(t, states[t, 1], '.r-', label='op_0', linewidth=0.3)
            elif options[t] == 2:
                ax.plot(t, states[t, 1], '.b-', label='op_1', linewidth=0.5)
            elif options[t] == 1:
                ax.plot(t, states[t, 1], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
            elif options[t] == 5:
                ax.plot(t, states[t, 1], '.c-', label='op_4', linewidth=0.7, alpha=0.4)
            elif options[t] == 4:
                ax.plot(t, states[t, 1], '.m-', label='op_5', linewidth=0.7, alpha=0.4)

            # Desired options
            # Trajectory x
            ax = fig.add_subplot(3, 4, 3)
            if des_options[t] == 0:
                ax.plot(t, states[t, 0], '.r-', label='dop_0', linewidth=0.3)
            elif des_options[t] == 2:
                ax.plot(t, states[t, 0], '.b-', label='dop_1', linewidth=0.5)
            elif des_options[t] == 1:
                ax.plot(t, states[t, 0], '.g-', label='dop_2', linewidth=0.7, alpha=0.2)
            elif des_options[t] == 5:
                ax.plot(t, states[t, 0], '.c-', label='dop_4', linewidth=0.7, alpha=0.4)
            elif des_options[t] == 4:
                ax.plot(t, states[t, 0], '.m-', label='dop_5', linewidth=0.7, alpha=0.4)

            # Trajectory Y
            ax = fig.add_subplot(3, 4, 4)
            if des_options[t] == 0:
                ax.plot(t, states[t, 1], '.r-', label='dop_0', linewidth=0.3)
            elif des_options[t] == 2:
                ax.plot(t, states[t, 1], '.b-', label='dop_1', linewidth=0.5)
            elif des_options[t] == 1:
                ax.plot(t, states[t, 1], '.g-', label='dop_2', linewidth=0.7, alpha=0.2)
            elif des_options[t] == 5:
                ax.plot(t, states[t, 1], '.c-', label='dop_4', linewidth=0.7, alpha=0.4)
            elif des_options[t] == 4:
                ax.plot(t, states[t, 1], '.m-', label='dop_5', linewidth=0.7, alpha=0.4)

            # Option Value
            ax = fig.add_subplot(3, 4, 5)
            ax.plot(t, value[t, 0], '.r-', label='op_0', linewidth=0.3)
            ax.plot(t, value[t, 2], '.b-', label='op_1', linewidth=0.5)
            ax.plot(t, value[t, 1], '.g-', label='op_2', linewidth=0.6, alpha=0.2)
            ax.plot(t, value[t, 5], '.c-', label='op_3', linewidth=0.7, alpha=0.4)
            ax.plot(t, value[t, 4], '.m-', label='op_3', linewidth=0.8, alpha=0.4)

            # Advantage
            ax = fig.add_subplot(3, 4, 6)
            ax.plot(t, adv[t], '.r-', label='op_0', linewidth=0.3)

            # Importance Sampling
            ax = fig.add_subplot(3, 4, 7)
            ax.plot(t, is_val[t], '.k-', label='op_0', linewidth=0.3)


            #Interest and term opt 1
            ax = fig.add_subplot(3, 4, 8)
            ax.plot(t, interest[t, 0], '.r-', label='op_0', linewidth=0.3)
            ax.plot(t, termination[t, 0], '.k-', label='op_0', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(3, 4, 9)
            ax.plot(t, interest[t, 2], '.b-', label='op_1', linewidth=0.3)
            ax.plot(t, termination[t, 2], '.k-', label='op_1', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(3, 4, 10)
            ax.plot(t, interest[t, 1], '.g-', label='op_2', linewidth=0.3)
            ax.plot(t, termination[t, 1], '.k-', label='op_2', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(3, 4, 11)
            ax.plot(t, interest[t, 5], '.c-', label='op_3', linewidth=0.3)
            ax.plot(t, termination[t, 5], '.k-', label='op_3', linewidth=0.5, alpha=0.2)

            ax = fig.add_subplot(3, 4, 12)
            ax.plot(t, interest[t, 4], '.m-', label='op_0', linewidth=0.5)
            ax.plot(t, termination[t, 4], '.k-', label='op_0', linewidth=0.5, alpha=0.2)



    file_name = 'figures/MOAC/isexp2_7/iteration_' + str(iter)
    plt.savefig(file_name)
    plt.close()


model = pickle.load(open('results/MOAC/isexp2_7/model/hybrid_model.pkl','rb'))

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
plt.savefig('figures/MOAC/isexp2_7/transition_graph.png')

print("Time taken:", time.time() - stime)
