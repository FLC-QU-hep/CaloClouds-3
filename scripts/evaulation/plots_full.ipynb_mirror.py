# %load_ext autoreload
# %autoreload 2

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

import importlib
import numpy as np
import pickle

from pointcloud.configs import Configs
import pointcloud.utils.plotting as plotting

cfg = Configs()

print(cfg.__dict__)
importlib.reload(plotting)

title = r'\textbf{full spectrum}' #  r'\textbf{50 GeV}'
real_label = r'\textsc{Geant4}'
ddpm_label = r'\textsc{CaloClouds}'
edm_label = r'\textsc{CaloClouds II}'
cm_label = r'\textsc{CaloClouds II (CM)}'
# # CoG
pickle_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/full/pickle/'
min_energy = 10
max_energy = 90

with open(pickle_path+'dict_{}-{}GeV.pickle'.format(str(min_energy), str(max_energy)), 'rb') as f:
    dict = pickle.load(f)

c_cog_real = dict['c_cog_real']
c_cog_fake = dict['c_cog_fake']
real_list = dict['real_list']
fakes_list = dict['fakes_list']
    
print('dicts loaded')
print(dict.keys())
# number of cell hits above threshold
real_list[-1][real_list[-1] >= 0.1].shape
importlib.reload(plotting)
plotting.plt_cog(c_cog_real, c_cog_fake, [real_label, ddpm_label, edm_label, cm_label], title=title)
# # other plots
24/3

importlib.reload(plotting)

plotting.get_plots_from_observables(real_list, fakes_list, labels = [real_label, ddpm_label, edm_label, cm_label], title=title, events=40_000)




