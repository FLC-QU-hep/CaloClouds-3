path = "/home/henry/training/point-cloud-diffusion-logs/investigations/data_loader_sample.npz"
import numpy as np
loaded = np.load(path)
max_events = 1000
e = loaded['e'][:max_events]
x = loaded['x'][:max_events]
energy = x[:, :, 3]
ys = x[:, :, 2]

mask = np.ones_like(energy, dtype=bool)
for i, en in enumerate(energy):
    trim_len = next(i for i in range(len(en)-1, 0, -1) if en[i] > 0)
    mask[i][trim_len:] = False
flat_masked_ys = ys.flatten()[mask.flatten()]
flat_masked_es = energy.flatten()[mask.flatten()]
from matplotlib import pyplot as plt
bins = np.linspace(-1, 1, 31)
plt.hist(flat_masked_ys, weights=flat_masked_es, bins=bins)
plt.semilogy()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for event in range(6):
    en = energy[event]
    
    y = ys[event]
    trim_len = next(i for i in range(len(en)-1, 0, -1) if en[i] > 0)
    y_values, bins, patches = ax1.hist(y[:trim_len], histtype='step', label='hits')
    e_values, bins, patches = ax2.hist(y[:trim_len], bins=bins, weights=en[:trim_len], histtype='step', label='energy')
    ax3.plot((bins[1:] + bins[:-1])*0.5, e_values/y_values, label='energy per hit')
    
en

