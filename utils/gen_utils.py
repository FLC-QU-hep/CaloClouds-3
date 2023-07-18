import numpy as np
import torch
from tqdm import tqdm

from utils.plotting import MAP, offset, layer_bottom_pos, cell_thickness, Xmax, Xmin, Zmax, Zmin



def get_cog(x,y,z,e):
    return np.sum((x * e), axis=1) / e.sum(axis=1), np.sum((y * e), axis=1) / e.sum(axis=1), np.sum((z * e), axis=1) / e.sum(axis=1)


def get_scale_factor(num_clusters, coef_real, coef_fake):
    
    # CaloClouds
    # coef_real = np.array([ 2.57988645e-09, -2.94056522e-05,  3.42194568e-01,  5.34968378e+01])
    # coef_fake = np.array([ 3.85057207e-09, -4.16463897e-05,  4.19800713e-01,  5.82246858e+01])

    # kCaloClouds_2023_05_24__14_54_09_heun18
    # coef_real = np.array([ 2.39735048e-09, -2.69842295e-05,  2.96136986e-01,  4.89770787e+01])
    # coef_fake = np.array([ 4.45753201e-09, -4.26483492e-05,  4.03632976e-01,  6.31063427e+01])

    # kCaloClouds_2023_05_24__14_54_09_heun13
    #coef_real = np.array([ 2.39735048e-09, -2.69842295e-05,  2.96136986e-01,  4.89770787e+01])
    #coef_fake = np.array([ 5.72940149e-09, -4.76120436e-05,  4.37720799e-01,  5.97962496e+01])
    
    poly_fn_real = np.poly1d(coef_real)
    poly_fn_fake = np.poly1d(coef_fake) 
    
    scale_factor = poly_fn_real(num_clusters) / poly_fn_fake(num_clusters)

    return scale_factor


def get_shower(model, num_points, energy, cond_N, bs=1, kdiffusion=False, config=None):
#     e = torch.ones((bs, 1), device=cfg.device) * float(energy)
#     n = torch.ones((bs, 1), device=cfg.device) * float(num_points)

    e = torch.ones((bs, 1), device=config.device) * energy
    n = torch.ones((bs, 1), device=config.device) * cond_N
    
    if config.norm_cond:
        e = e / 100 * 2 -1   # max incident energy: 100 GeV
        n = n / config.max_points * 2  - 1
    cond_feats = torch.cat([e, n], -1)
        
    with torch.no_grad():
        if kdiffusion:
            fake_shower = model.sample(cond_feats, num_points, config)
        else:
            fake_shower = model.sample(cond_feats, num_points, config.flexibility)
    
    return fake_shower


# batch inference 
def gen_showers_batch(model, shower_flow, e_min, e_max, num=2000, bs=32, kdiffusion=False, config=None, max_points=6000, coef_real=None, coef_fake=None, n_scaling = True):
    
    cond_E = torch.FloatTensor(num, 1).uniform_(e_min, e_max).to(config.device)
    samples = shower_flow.condition(cond_E/100).sample(torch.Size([num, ])).cpu().numpy()

    num_clusters = np.clip((samples[:, 0] * 5000).reshape(num, 1), 0, max_points)
    energies = np.clip((samples[:, 1] * 2.5 * 1000).reshape(num, 1), 0, None)
    cog_x = samples[:, 2] * 25
    cog_y = samples[:, 3] * 15 + 15
    cog_z = samples[:, 4] * 20 + 40

    # clusters_per_layer_gen = samples[:, 2:32] * 400
    # clusters_per_layer_gen[clusters_per_layer_gen < 0] = 0
    clusters_per_layer_gen = np.clip(samples[:, 5:35], 0, 1)  #  B,30
    e_per_layer_gen = np.clip(samples[:, 35:], 0, 1)          #  B,30

    # resample if sum of clusters larger than max_points   
    # while np.any(clusters_per_layer_gen.sum(axis=1) > max_points):
    #     samples = distribution.condition(cond_E/100).sample(torch.Size([num, ])).cpu().numpy()
    #     clusters_per_layer_gen = samples[:, 2:32] * 400
    #     clusters_per_layer_gen[clusters_per_layer_gen < 0] = 0

    # cog_x = samples[:, 32]
    # cog_y = samples[:, 33] + 15
    # cog_z = samples[:, 34] + 40

    # rescale number of clusters with polybomial function
    # scale_factor = get_scale_factor(clusters_per_layer_gen.sum(axis=1))   # B,
    # scale_factor = np.expand_dims(scale_factor, axis=1)
    if n_scaling:
        scale_factor = get_scale_factor(num_clusters, coef_real, coef_fake)   # B,1
        num_clusters = (num_clusters * scale_factor).astype(int)  # B,1
    else:
        num_clusters = (num_clusters).astype(int)  # B,1
    
    # scale relative clusters per layer to actual number of clusters per layer
    clusters_per_layer_gen = (clusters_per_layer_gen / clusters_per_layer_gen.sum(axis=1, keepdims=True) * num_clusters).astype(int) # B,30
    # same for energy
    e_per_layer_gen = e_per_layer_gen / e_per_layer_gen.sum(axis=1, keepdims=True) * energies  # B,30

    #clusters_per_layer_gen = (clusters_per_layer_gen * scale_factor).astype(int)

    # clusters_per_layer_gen = (clusters_per_layer_gen).astype(int)

    # sort by number of clusters for better batching and faster inference
    mask = np.argsort(num_clusters.squeeze())
    #mask = np.argsort(clusters_per_layer_gen.sum(axis=1))
    cond_E = cond_E[mask]
    num_clusters = num_clusters[mask]
    energies = energies[mask]
    cog_x = cog_x[mask]
    cog_y = cog_y[mask]
    cog_z = cog_z[mask]
    clusters_per_layer_gen = clusters_per_layer_gen[mask]
    e_per_layer_gen = e_per_layer_gen[mask]

    fake_showers_list = []
    
    for evt_id in tqdm(range(0, num, bs), disable=False):
        if (num - evt_id) < bs:
            bs = num - evt_id
#         num = num - 75 # substruct correction factor
        # convert clusters_per_layer_gen to a fractions of points in the layer out of sum(points in the layer) of event
        # multuply clusters_per_layer_gen by corrected tottal num of points
        hits_per_layer_all = clusters_per_layer_gen[evt_id : evt_id+bs] # shape (bs, num_layers) 
        e_per_layer_all = e_per_layer_gen[evt_id : evt_id+bs] # shape (bs, num_layers)
        max_num_clusters = hits_per_layer_all.sum(axis=1).max()
        cond_N = torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)
        
        fs = get_shower(model, max_num_clusters, cond_E[evt_id : evt_id+bs], cond_N, bs=bs, kdiffusion=kdiffusion, config=config)
        fs = fs.cpu().numpy()
        
#         if np.isnan(fs).sum() != 0:
# #             return fs
#             print('nans in showers!')
#             fs[np.isnan(fs)] = 0
# #             break
        
        # loop over events
        y_positions = layer_bottom_pos+cell_thickness/2
        for i, hits_per_layer in enumerate(hits_per_layer_all):
        #for i, (hits_per_layer, e_per_layer) in enumerate(zip(hits_per_layer_all, e_per_layer_all)):
    
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

            y_flow = np.repeat(y_positions, hits_per_layer)
            y_flow = np.concatenate([y_flow, np.zeros(n_hits_to_concat)])

            mask = np.concatenate([ np.ones( hits_per_layer.sum() ), np.zeros( n_hits_to_concat ) ])

            fs[i, :, 1][mask == 0] = 10
            idx_dm = np.argsort(fs[i, :, 1])
            fs[i, :, 1][idx_dm] = y_flow


            fs[i, :, :][y_flow==0] = 0  

        # fs[:, :, -1] = abs(fs[:, :, -1])  
            fs[fs[:, :, -1]  <= 0] = 0    # setting events with negative energy to zero
        #fs[:, :, -1] = fs[:, :, -1] / fs[:, :, -1].sum(axis=1).reshape(bs, 1) * energies[evt_id : evt_id+bs] # energy rescaling to predicted e_sum

            # energy per layer calibration
            for j in range(len(y_positions)):
                mask = fs[i, :, 1] == y_positions[j]
                fs[i, :, -1][mask] = fs[i, :, -1][mask] / fs[i, :, -1][mask].sum() * e_per_layer_all[i, j]

        length = max_points - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)
        fake_showers_list.append(fs)
        
    fake_showers = np.vstack(fake_showers_list)  # (bs, num_points, 4)

    # # energy per layer calibration
    # y_positions = np.unique(fake_showers[:, :, 1])[1:]   # y_positions without zero
    # # loop over layers
    # for i in range(len(y_positions)):  
    #     # loop over events
    #     for j in range(len(fake_showers)):  
    #         # get indices of points in layer
    #         idx = np.where(fake_showers[j, :, 1] == y_positions[i])[0]  
    #         # set energy of layer
    #         fake_showers[j, idx, -1] = fake_showers[j, idx, -1] / fake_showers[j, idx, -1].sum() * e_per_layer_gen[j, i]

    
    fake_showers = np.moveaxis(fake_showers, -1, -2)   # (bs, num_points, 4) -> (bs, 4, num_points)
    fake_showers[:, 0, :] = (fake_showers[:, 0, :] + 1) / 2
    fake_showers[:, 2, :] = (fake_showers[:, 2, :] + 1) / 2
    
    fake_showers[:, 0] = fake_showers[:, 0] * (Xmin-Xmax) + Xmax
    fake_showers[:, 2] = fake_showers[:, 2] * (Zmin-Zmax) + Zmax

    # CoG calibration
    cog = get_cog(fake_showers[:, 0], fake_showers[:, 1], fake_showers[:, 2], fake_showers[:, 3])
    fake_showers[:, 0] -= (cog[0] - cog_x)[:,None]
    fake_showers[:, 2] -= (cog[2] - cog_z)[:,None]
    
    return fake_showers