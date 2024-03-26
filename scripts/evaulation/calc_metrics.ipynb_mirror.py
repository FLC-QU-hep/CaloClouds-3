import sys

sys.path.append('../')



import numpy as np

import matplotlib.pyplot as plt

import importlib

import pickle

from scipy.stats import wasserstein_distance



import pointcloud.utils.metrics as metrics
pickle_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/metrics/'



dict_real = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_g4.pickle', 'rb'))

dict_ddpm = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_ddpm.pickle', 'rb'))

dict_edm = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_edm.pickle', 'rb'))

dict_cm = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_cm.pickle', 'rb'))



print(dict_cm.keys())
# combine observables in a single array



obs_real = metrics.get_event_observables_from_dict(dict_real)

obs_ddpm = metrics.get_event_observables_from_dict(dict_ddpm)

obs_edm = metrics.get_event_observables_from_dict(dict_edm)

obs_cm = metrics.get_event_observables_from_dict(dict_cm)



print(obs_real.shape)



mean_real, std_real = np.mean(obs_real, axis=0).reshape(1,-1), np.std(obs_real, axis=0).reshape(1,-1)



print(mean_real.shape)



# shuffle the observables, since during generation they were ordered by number of clusters (in their respective chunks)

np.random.seed(42)

mask_ddpm = np.random.permutation(len(obs_ddpm))

mask_edm = np.random.permutation(len(obs_edm))

mask_cm = np.random.permutation(len(obs_cm))

obs_ddpm = obs_ddpm[mask_ddpm]

obs_edm = obs_edm[mask_edm]

obs_cm = obs_cm[mask_cm]

obs_ddpm[20000:20010, 3]
# standardise the data

def standardize(ary, mean, std):

    return (ary - mean) / std



obs_std_real = standardize(obs_real, mean=mean_real, std=std_real)

obs_std_ddpm = standardize(obs_ddpm, mean=mean_real, std=std_real)

obs_std_edm = standardize(obs_edm, mean=mean_real, std=std_real)

obs_std_cm = standardize(obs_cm, mean=mean_real, std=std_real)
# array without hits

obs_std_real_woutHits = np.concatenate([obs_std_real[:,0:5], obs_std_real[:,6:]], axis=1)

obs_std_ddpm_woutHits = np.concatenate([obs_std_ddpm[:,0:5], obs_std_ddpm[:,6:]], axis=1)

obs_std_edm_woutHits = np.concatenate([obs_std_edm[:,0:5], obs_std_edm[:,6:]], axis=1)

obs_std_cm_woutHits = np.concatenate([obs_std_cm[:,0:5], obs_std_cm[:,6:]], axis=1)



print(obs_std_real_woutHits.shape)
# plot all features



# for i in range(obs_std_real.shape[1]):

#     h = plt.hist(obs_std_real[:,i], bins=50, alpha=0.5, label='g4')

#     plt.hist(obs_std_ddpm[:,i], bins=h[1], label='ddpm', histtype='step')

#     plt.hist(obs_std_edm[:,i], bins=h[1], label='edm', histtype='step')

#     plt.hist(obs_std_cm[:,i], bins=h[1], label='cm', histtype='step')

#     plt.legend(loc='best')

#     plt.xlabel('feature {}'.format(i))

#     plt.yscale('log')

#     plt.show()
importlib.reload(metrics)



means, stds = metrics.calc_wdist(obs_std_real, obs_std_ddpm, iterations=10, batch_size=50_000)

means *= 100

stds *= 100



print(means.shape)



print('scores for ddpm: ')

print('occ: ' + str(means[3].round(1)) + ' $\pm$ ' + str(stds[3].round(1)))

print('sampling_fraction: ' + str(means[4].round(1)) + ' $\pm$ ' + str(stds[4].round(1)))

print('hits: ' + str(means[5].round(1)) + ' $\pm$ ' + str(stds[5].round(1)))



mean, std = metrics.combine_scores(means[6:16], stds[6:16])

print('binned_layer_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



mean, std = metrics.combine_scores(means[16:26], stds[16:26])

print('binned_radial_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



print('cog_x: ' + str(means[0].round(1)) + ' $\pm$ ' + str(stds[0].round(1)))

print('cog_y: ' + str(means[2].round(1)) + ' $\pm$ ' + str(stds[2].round(1)))

print('cog_z: ' + str(means[1].round(1)) + ' $\pm$ ' + str(stds[1].round(1)))
means, stds = metrics.calc_wdist(obs_std_real, obs_std_edm, iterations=10, batch_size=50_000)

means *= 100

stds *= 100



print(means.shape)



print('scores for edm: ')

print('occ: ' + str(means[3].round(1)) + ' $\pm$ ' + str(stds[3].round(1)))

print('sampling_fraction: ' + str(means[4].round(1)) + ' $\pm$ ' + str(stds[4].round(1)))

print('hits: ' + str(means[5].round(1)) + ' $\pm$ ' + str(stds[5].round(1)))



mean, std = metrics.combine_scores(means[6:16], stds[6:16])

print('binned_layer_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



mean, std = metrics.combine_scores(means[16:26], stds[16:26])

print('binned_radial_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



print('cog_x: ' + str(means[0].round(1)) + ' $\pm$ ' + str(stds[0].round(1)))

print('cog_y: ' + str(means[2].round(1)) + ' $\pm$ ' + str(stds[2].round(1)))

print('cog_z: ' + str(means[1].round(1)) + ' $\pm$ ' + str(stds[1].round(1)))
means, stds = metrics.calc_wdist(obs_std_real, obs_std_cm, iterations=10, batch_size=50_000)

means *= 100

stds *= 100



print(means.shape)



print('scores for cm: ')

print('occ: ' + str(means[3].round(1)) + ' $\pm$ ' + str(stds[3].round(1)))

print('sampling_fraction: ' + str(means[4].round(1)) + ' $\pm$ ' + str(stds[4].round(1)))

print('hits: ' + str(means[5].round(1)) + ' $\pm$ ' + str(stds[5].round(1)))



mean, std = metrics.combine_scores(means[6:16], stds[6:16])

print('binned_layer_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



mean, std = metrics.combine_scores(means[16:26], stds[16:26])

print('binned_radial_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



print('cog_x: ' + str(means[0].round(1)) + ' $\pm$ ' + str(stds[0].round(1)))

print('cog_y: ' + str(means[2].round(1)) + ' $\pm$ ' + str(stds[2].round(1)))

print('cog_z: ' + str(means[1].round(1)) + ' $\pm$ ' + str(stds[1].round(1)))
importlib.reload(metrics)

means, stds = metrics.calc_wdist(obs_std_real, obs_std_real[::-1], iterations=10, batch_size=50_000)

means *= 100

stds *= 100



print(means.shape)



print('scores for truth (Geant4 with itself reversed (so no event is paird with itself)): ')

print('occ: ' + str(means[3].round(1)) + ' $\pm$ ' + str(stds[3].round(1)))

print('sampling_fraction: ' + str(means[4].round(1)) + ' $\pm$ ' + str(stds[4].round(1)))

print('hits: ' + str(means[5].round(1)) + ' $\pm$ ' + str(stds[5].round(1)))



mean, std = metrics.combine_scores(means[6:16], stds[6:16])

print('binned_layer_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



mean, std = metrics.combine_scores(means[16:26], stds[16:26])

print('binned_radial_e: ' + str(mean.round(1)) + ' $\pm$ ' + str(std.round(1)))



print('cog_x: ' + str(means[0].round(1)) + ' $\pm$ ' + str(stds[0].round(1)))

print('cog_y: ' + str(means[2].round(1)) + ' $\pm$ ' + str(stds[2].round(1)))

print('cog_z: ' + str(means[1].round(1)) + ' $\pm$ ' + str(stds[1].round(1)))


