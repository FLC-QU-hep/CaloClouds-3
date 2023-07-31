# import sys
# sys.path.append('../')
# import time
# s_time = time.time()

# import h5py
# import numpy as np
# import sys
# import pickle
# import h5py

# from configs import Configs
# import utils.plotting as plotting
# from utils.plotting import get_projections, MAP, layer_bottom_pos

# cfg = Configs()

# # print(cfg.__dict__)





# # number of events to be processed
# n = 40000   # default 40k
# pickle_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/pickle/'






# #### load save data

# # path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/validation/50GeV_x36_grid_regular_2k_Z4.hdf5'
# # path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/validation/10-90GeV_x36_grid_regular_float32.hdf5'
# # path = '/beegfs/desy/user/akorol/projects/getting_high/ILDConfig/StandardConfig/production/out/10GeV_x36_grid_regular_2k_Z4_grid_pos_rundom.hdf5'
# path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/all_steps/validation/photon-showers_10-90GeV_A90_Zpos4.slcio.hdf5'
# real_showers = h5py.File(path, 'r')['events'][:]
# real_showers[:, -1] = real_showers[:, -1] * 1000   # GeV to MeV
# print('real showers shape: ', real_showers.shape)

# # fake_showers = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/CCbaseline_50GeV_2000.npy')
# fake_showers = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/CCbaseline_10-90GeV_40k_wNscaling.npy')
# print('fake showers shape: ', fake_showers.shape)

# # fake_showers_2 = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/kCaloClouds_2023_05_24__14_54_09_heun18_50GeV_2k.npy')
# # fake_showers_2 = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/kCaloClouds_2023_05_24__14_54_09_heun13_50GeV_2k.npy')
# # fake_showers_2 = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/kCaloClouds_2023_06_29__23_08_31_ckpt_0.000000_2000000_heun13_50GeV_2k.npy')
# fake_showers_2 = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/kCaloClouds_2023_06_29__23_08_31_ckpt_0.000000_2000000_10-90GeV_40k_wNscaling.npy')
# print('fake showers shape: ', fake_showers_2.shape)

# # fake_showers_3 = np.load('/beegfs/desy/user/akorol/projects/point-cloud/DM_new_100s_90GeV_with_flow_corrections_3.npy')
# fake_showers_3 = np.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/CD_2023_07_07__16_32_09_ckpt_0.000000_1000000_10-90GeV_40k_wNscaling.npy')
# print('fake showers shape: ', fake_showers_3.shape)


# # projection
# print('files loaded (all energy hits in MeV). now projection.')
# events, cloud = get_projections(real_showers[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)
# events_fake, cloud_fake = get_projections(fake_showers[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)
# events_fake_2, cloud_fake_2 = get_projections(fake_showers_2[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)
# events_fake_3, cloud_fake_3 = get_projections(fake_showers_3[:n], MAP, layer_bottom_pos, max_num_hits=6000, return_cell_point_cloud=True)

# # calculate cog
# c_cog = plotting.get_cog(cloud)
# print('len c_cog: ', len(c_cog[0]), len(c_cog[1]), len(c_cog[2]))
# c_cog_2 = plotting.get_cog(cloud_fake)
# print('len c_cog_2: ', len(c_cog_2[0]), len(c_cog_2[1]), len(c_cog_2[2]))
# c_cog_3 = plotting.get_cog(cloud_fake_2)
# print('len c_cog_3: ', len(c_cog_3[0]), len(c_cog_3[1]), len(c_cog_3[2]))
# c_cog_4 = plotting.get_cog(cloud_fake_3)
# print('len c_cog_4: ', len(c_cog_4[0]), len(c_cog_4[1]), len(c_cog_4[2]))
# c_cog_real = c_cog
# c_cog_fake = [c_cog_2, c_cog_3, c_cog_4]

# with open(pickle_path+'c_cog_real.pickle', 'wb') as f:
#     pickle.dump(c_cog_real, f)
# with open(pickle_path+'c_cog_fake.pickle', 'wb') as f:
#     pickle.dump(c_cog_fake, f)
# print('cog saved in pickle files')


# # features / observablesc
# print('get observables')
# real_list, fakes_list = plotting.get_observables_for_plotting(events, [events_fake, events_fake_2, events_fake_3])

# with open(pickle_path+'real_list.pickle', 'wb') as f:
#     pickle.dump(real_list, f)
# with open(pickle_path+'fakes_list.pickle', 'wb') as f:
#     pickle.dump(fakes_list, f)
# print('observalbes saved in pickle files')

# print('done. took {} mins'.format((time.time()-s_time)/60))