import numpy as np
import pickle
import matplotlib.pyplot as plt
from common.model_learning_utils import scale
import time
from model_learning import partialHybridModel
import networkx as nx
import os
# supressing warnings
import warnings
import matplotlib.cbook
from xlwt import Workbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

stime = time.time()
f = open("results/MOAC/blckIn_exp4_A1/data/rollout_data.pkl", "rb")
p = pickle.load(f)
f.close()

model = pickle.load(open('results/MOAC/blckIn_exp4/model/hybrid_model.pkl', 'rb'))

print("Model graph:", model.transitionGraph.nodes)
print("Model options:", model.transitionGraph.edges)
edges = list(model.transitionGraph.edges)
for i in range(0, len(edges)):
    print(edges[i][0], " -> ", edges[i][1], " : ", model.transitionGraph[edges[i][0]][edges[i][1]]['weight'])
#G = model.transitionGraph
#pos = nx.spring_layout(G)
#nx.draw(G, pos, with_labels=True)
#plt.savefig('TGraph_blcInsert.png')
#plt.clf()

'''
nx = 100
ny = 150

x = np.linspace(-0.5, 0.25, nx)
y = np.linspace(0, 0.9, ny)
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
ob = []
for i in range(0, nx):
    for j in range(0, ny):
        ob.append([xv[i, j], yv[i, j]])

ob = np.array(ob)
print(ob.shape)
z = model.modeFunction.predict_f(ob)
zv = np.zeros_like(xv)
mask = np.zeros_like(zv, dtype=bool)

count = 0
for i in range(0, nx):
    for j in range(0, ny):
        zv[i, j] = z[count, 1]
        Mode_11 = [-0.05, 0.2, 0.25, 0.75]
        Mode_12 = [-0.05, 0.05, 0.45, 0.55]
        point = [xv[i, j], yv[i, j]]
        if Mode_11[0] <= point[0] <= Mode_11[1] and Mode_11[2] <= point[1] <= Mode_11[3]:
            mask[i, j] = True
        if Mode_12[0] <= point[0] <= Mode_12[1] and Mode_12[2] <= point[1] <= Mode_12[3]:
            mask[i, j] = False
        count += 1
z_masked = np.ma.array(zv, mask=mask)

print(z.shape)
plt.contourf(xv, yv, z_masked)
plt.colorbar()
plt.xlabel('X axis', fontsize=15)
plt.ylabel('Y axis', fontsize=15)
figure = plt.gcf()
figure.set_size_inches(9, 8)
plt.savefig('ModePredict0.png', dpi=500)

nx = 100
ny = 100

x = np.linspace(0, 1.2, nx)
y = np.linspace(0, 1.2, ny)
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
ob = []
for i in range(0, nx):
    for j in range(0, ny):
        ob.append([xv[i, j], yv[i, j]])

ob = np.array(ob)
print(ob.shape)
z = model.modeFunction.predict_f(ob)
z_scaled = scale(z, 0, 1)
print(z_scaled)
zv0 = np.zeros_like(xv)
zv1 = np.zeros_like(xv)
zv2 = np.zeros_like(xv)

count = 0
for i in range(0, nx):
    for j in range(0, ny):
        zv0[i, j] = z_scaled[count, 0]
        zv1[i, j] = z_scaled[count, 1]
        zv2[i, j] = z_scaled[count, 2]
        count = count+1

print(z.shape)
plt.contourf(xv, yv, zv0)
plt.colorbar()
plt.xlabel('X axis', fontsize=15)
plt.ylabel('Y axis', fontsize=15)
figure = plt.gcf()
figure.set_size_inches(9, 8)
plt.savefig('ModePredict0.png', dpi=500)

plt.clf()
plt.contourf(xv, yv, zv1)
plt.colorbar()
plt.xlabel('X axis', fontsize=15)
plt.ylabel('Y axis', fontsize=15)
figure = plt.gcf()
figure.set_size_inches(9, 8)
plt.savefig('ModePredict1.png', dpi=500)

plt.clf()
plt.contourf(xv, yv, zv2)
plt.colorbar()
plt.xlabel('X axis', fontsize=15)
plt.ylabel('Y axis', fontsize=15)
figure = plt.gcf()
figure.set_size_inches(9, 8)
plt.savefig('ModePredict2.png', dpi=500)

#data = {'x': xv, 'y': yv, 'z': z, 'z0': zv0, 'z1': zv1, 'z2': zv2}
#pickle.dump(data, open("modeclassifier.pkl", "wb"))
'''
#iteration = 25
horizon = 80
rolloutSize = 100
selected_options = [0, 2, 1, 3]
main_path = 'figures/MOAC/OptionsAnalysis/blcIns_exp4_A1'
wb = Workbook()
sheet = wb.add_sheet('Final Data check')

data = p[0]
rollouts = data['rollouts']
model.learnPreDefModes(rollouts)
segment_data = model.segment_data
pickle.dump(segment_data, open("blcIns_seg_data_A1.pkl", "wb"))



'''
rolloutList = [5, 10, 13, 16, 77, 77]
segs_data = []


for i in range(0, len(rolloutList)):
    data = p[rolloutList[i]]
    rollouts = data['rollouts']
    model.learnPreDefModes(rollouts)
    segment_data = model.segment_data
    segs_data.append(segment_data)

sd = {'sdata': segs_data}
pickle.dump(sd, open("blcIns_seg_data.pkl", "wb"))
'''
for iter in range(0, len(p)):
    iter_path = main_path+'/iter_'+str(iter)
    os.mkdir(iter_path)
    print("Iteration : ", iter)
    data = p[iter]
    rollouts = data['rollouts']
    model.learnPreDefModes(rollouts)
    segment_data = model.segment_data
    obs = rollouts['ob']
    acs = rollouts['ac']
    opts = rollouts['opts']
    activ_opt = rollouts['activated_options']

    seg_sample_count = 0

    #segmented variables
    vpred = rollouts['vpreds']
    des_opts = rollouts['des_opts']
    term = rollouts['betas']
    advs = rollouts["adv"]
    is_values = rollouts['is']
    term = rollouts['betas']
    for rollout in range(0, rolloutSize-1):
        # Prepare for plotting
        fig = plt.figure()
        #print("Rollout: ", rollout)
        states = obs[rollout * horizon:(horizon + rollout * horizon), :]
        action = acs[rollout * horizon:(horizon + rollout * horizon), :]
        options = opts[rollout * horizon:(horizon + rollout * horizon)]
        interest = activ_opt[rollout * horizon:(horizon + rollout * horizon)]
        num_segments = len(segment_data[rollout][1])

        similarity = 0
        prev_time = 0
        for segment in range(0, num_segments):
            segStates = states[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1] + 1), :]
            segAction = action[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1] + 1), :]
            segOpts = options[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1] + 1)]
            segInt = interest[segment_data[rollout][1][segment][0]:(segment_data[rollout][1][segment][1] + 1)]
            segTime = len(segStates)

            for t in range(0, segTime):
                traj_time = prev_time + t
                if segOpts[t] == des_opts[seg_sample_count]:
                    similarity += 1

                # Selected options
                # Trajectory X
                ax = fig.add_subplot(3, 4, 1)
                if options[t] == selected_options[0]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.r-', label='op_0', linewidth=0.3)
                elif options[t] == selected_options[1]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.b-', label='op_1', linewidth=0.5)
                elif options[t] == selected_options[2]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
                elif options[t] == selected_options[3]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.c-', label='op_4', linewidth=0.7, alpha=0.4)
                #elif options[t] == selected_options[4]:
                #   ax.plot(traj_time, segStates[t, 0], '.m-', label='op_5', linewidth=0.7, alpha=0.4)

                '''
                # Trajectory Y
                ax = fig.add_subplot(3, 4, 2)
                if options[t] == selected_options[0]:
                    ax.plot(traj_time, segStates[t, 1], '.r-', label='op_0', linewidth=0.3)
                elif options[t] == selected_options[1]:
                    ax.plot(traj_time, segStates[t, 1], '.b-', label='op_1', linewidth=0.5)
                elif options[t] == selected_options[2]:
                    ax.plot(traj_time, segStates[t, 1], '.g-', label='op_2', linewidth=0.7, alpha=0.2)
                elif options[t] == selected_options[3]:
                    ax.plot(traj_time, segStates[t, 1], '.c-', label='op_4', linewidth=0.7, alpha=0.4)
                #elif options[t] == selected_options[4]:
                #    ax.plot(traj_time, segStates[t, 1], '.m-', label='op_5', linewidth=0.7, alpha=0.4)
                '''
                # Desired options
                # Trajectory x
                ax = fig.add_subplot(3, 4, 3)
                if des_opts[seg_sample_count] == selected_options[0]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.r-', label='dop_0', linewidth=0.3)
                elif des_opts[seg_sample_count] == selected_options[1]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.b-', label='dop_1', linewidth=0.5)
                elif des_opts[seg_sample_count] == selected_options[2]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.g-', label='dop_2', linewidth=0.7, alpha=0.2)
                elif des_opts[seg_sample_count] == selected_options[3]:
                    ax.plot(segStates[t, 0], segStates[t, 1], '.c-', label='dop_4', linewidth=0.7, alpha=0.4)
                #elif des_opts[seg_sample_count] == selected_options[4]:
                #    ax.plot(traj_time, segStates[t, 0], '.m-', label='dop_5', linewidth=0.7, alpha=0.4)
                '''
                # Trajectory Y
                ax = fig.add_subplot(3, 4, 4)
                if des_opts[seg_sample_count] == selected_options[0]:
                    ax.plot(traj_time, segStates[t, 1], '.r-', label='dop_0', linewidth=0.3)
                elif des_opts[seg_sample_count] == selected_options[1]:
                    ax.plot(traj_time, segStates[t, 1], '.b-', label='dop_1', linewidth=0.5)
                elif des_opts[seg_sample_count] == selected_options[2]:
                    ax.plot(traj_time, segStates[t, 1], '.g-', label='dop_2', linewidth=0.7, alpha=0.2)
                elif des_opts[seg_sample_count] == selected_options[3]:
                    ax.plot(traj_time, segStates[t, 1], '.c-', label='dop_4', linewidth=0.7, alpha=0.4)
                #elif des_opts[seg_sample_count] == selected_options[4]:
                #    ax.plot(traj_time, segStates[t, 1], '.m-', label='dop_5', linewidth=0.7, alpha=0.4)
                '''
                # Option Value
                ax = fig.add_subplot(3, 4, 5)
                ax.plot(traj_time, vpred[seg_sample_count, selected_options[0]], '.r-', label='op_0', linewidth=0.3)
                ax.plot(traj_time, vpred[seg_sample_count, selected_options[1]], '.b-', label='op_1', linewidth=0.5)
                ax.plot(traj_time, vpred[seg_sample_count, selected_options[2]], '.g-', label='op_2', linewidth=0.6, alpha=0.2)
                ax.plot(traj_time, vpred[seg_sample_count, selected_options[3]], '.c-', label='op_3', linewidth=0.7, alpha=0.4)
                #ax.plot(traj_time, vpred[seg_sample_count, selected_options[4]], '.m-', label='op_3', linewidth=0.8, alpha=0.4)

                # Advantage
                ax = fig.add_subplot(3, 4, 6)
                ax.plot(traj_time, advs[seg_sample_count], '.r-', label='op_0', linewidth=0.3)

                # Importance Sampling
                ax = fig.add_subplot(3, 4, 7)
                ax.plot(traj_time, is_values[seg_sample_count], '.k-', label='op_0', linewidth=0.3)


                #Interest and term opt 1
                ax = fig.add_subplot(3, 4, 8)
                ax.plot(traj_time, segInt[t, selected_options[0]], '.r-', label='op_0', linewidth=0.3)
                ax.plot(traj_time, term[seg_sample_count, selected_options[0]], '.k-', label='op_0', linewidth=0.5, alpha=0.2)

                ax = fig.add_subplot(3, 4, 9)
                ax.plot(traj_time, segInt[t, selected_options[1]], '.b-', label='op_1', linewidth=0.3)
                ax.plot(traj_time, term[seg_sample_count, selected_options[1]], '.k-', label='op_1', linewidth=0.5, alpha=0.2)

                ax = fig.add_subplot(3, 4, 10)
                ax.plot(traj_time, segInt[t, selected_options[2]], '.g-', label='op_2', linewidth=0.3)
                ax.plot(traj_time, term[seg_sample_count, selected_options[2]], '.k-', label='op_2', linewidth=0.5, alpha=0.2)

                ax = fig.add_subplot(3, 4, 11)
                ax.plot(traj_time, segInt[t, selected_options[3]], '.c-', label='op_3', linewidth=0.3)
                ax.plot(traj_time, term[seg_sample_count, selected_options[3]], '.k-', label='op_3', linewidth=0.5, alpha=0.2)

                #ax = fig.add_subplot(3, 4, 12)
                #ax.plot(traj_time, segInt[t, selected_options[4]], '.m-', label='op_0', linewidth=0.5)
                #ax.plot(traj_time, term[seg_sample_count, selected_options[4]], '.k-', label='op_0', linewidth=0.5, alpha=0.2)

                seg_sample_count += 1
            prev_time += segTime

        file_name = iter_path+'/rollout_' + str(rollout)
        plt.savefig(file_name)
        plt.close()

        sheet.write(iter, rollout, str(similarity/80))


wb.save(main_path+'/similarity_check.xls')
print("Time taken:", time.time() - stime)
