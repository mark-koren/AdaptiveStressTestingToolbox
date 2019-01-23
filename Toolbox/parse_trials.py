import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = './data/lr_64/'
filenames = ['l1/tab.txt', 'l2/tab.txt', 'l3/tab.txt', 'l4/tab.txt', 'l5/tab.txt', 'l6/tab.txt', 'l7/tab.txt', 'l8/tab.txt', 'l9/tab.txt']
ax = None
for f in filenames:
    data = pd.read_csv(filepath + f)
    d_cumMaxAvg = data.assign(CumAvgReturn=data['AverageReturn'].cummax())
    ax = d_cumMaxAvg.plot('Iteration','CumAvgReturn', ax=ax, legend=filenames)
plt.show()
ax = None
for f in filenames:
    data = pd.read_csv(filepath + f)
    d_cumMaxMax = data.assign(CumMaxReturn=data['MaxReturn'].cummax())
    ax = d_cumMaxMax.plot('Iteration','CumMaxReturn', ax=ax, legend=filenames)
plt.show()
ax = None
for f in filenames:
    data = pd.read_csv(filepath + f)
    ax = data.plot('Iteration','AverageReturn', ax=ax, legend=filenames)
plt.show()
ax = None
for f in filenames:
    data = pd.read_csv(filepath + f)
    ax = data.plot('Iteration','MaxReturn', ax=ax, legend=filenames)
plt.show()



# pdb.set_trace()
# # def save_trials():
# discrete_grid_world = np.zeros((1000,1000))
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('scripts/test_model.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('scripts/'))
#     graph = tf.get_default_graph()
#     with tf.variable_scope('Loader'):
#         data = joblib.load('data/debug1/run1' + '/itr_' + str(0) + '.pkl')
#         # pdb.set_trace()
#         paths = data['paths']
#         env = data['env']
#         policy = data['policy']
#         for i in range(0,5000):
#             r_out = rollout(env, policy)
#             obs = r_out['observations']
#             H, xe, ye = np.histogram2d(obs[:,2], obs[:,3], bins=1000, range=[[-50, 50], [-50, 50]])
#             discrete_grid_world += H
#             # pdb.set_trace()
#
#         np.savetxt(fname='.' + '/grid_' + str(0) + '.csv',
#                            X=discrete_grid_world,
#                            delimiter = ',')
        #sess.run(tf.global_variables_initializer())
    # for i in range(0, iters):
    #     if (np.mod(i, save_every_n) != 0):
    #         continue
    #     with tf.variable_scope('Loader' + str(i)):
    #         data = joblib.load(path + '/itr_' + str(i) + '.pkl')
    #         paths = data['paths']
    #         pdb.set_trace()
    #         trials = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
    #         crashes = np.array([]).reshape(0, paths[0]['env_infos']['info']['cache'].shape[1])
    #         for n, a_path in enumerate(paths):
    #             cache = a_path['env_infos']['info']['cache']
    #             cache[:, 0] = n
    #             trials = np.concatenate((trials, cache), axis=0)
    #             if cache[-1,-1] == 0.0:
    #                 crashes = np.concatenate((crashes, cache), axis=0)
    #
    #         np.savetxt(fname=path + '/trials_' + str(i) + '.csv',
    #                    X=trials,
    #                    delimiter=',',
    #                    header=header)
    #
    #         np.savetxt(fname=path + '/crashes_' + str(i) + '.csv',
    #                    X=crashes,
    #                    delimiter=',',
    #                    header=header)