
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

class Configs():
    
    def __init__(self):

    # legend font
        self.font = font_manager.FontProperties(
            family='serif',
            size=15
        )
        
    # radial profile
        self.bins_r = 35

    # occupancy
        self.occup_bins = np.linspace(200, 1500, 70)

    # hits
        self.hit_bins = np.logspace(np.log10(0.01), np.log10(200), 70)

    #CoG
        self.bins_cog = 50

cfg = Configs()

Ymin = 1811
Xmin = -200
Xmax = 200
Zmin = -160
Zmax = 240

half_cell_size = 5.0883331298828125/2
cell_thickness = 0.5250244140625

layer_bottom_pos = np.array([   1811.34020996, 1814.46508789, 1823.81005859, 1826.93505859,
                                    1836.2800293 , 1839.4050293 , 1848.75      , 1851.875     ,
                                    1861.2199707 , 1864.3449707 , 1873.68994141, 1876.81494141,
                                    1886.16003418, 1889.28503418, 1898.63000488, 1901.75500488,
                                    1911.09997559, 1914.22497559, 1923.56994629, 1926.69494629,
                                    1938.14001465, 1943.36499023, 1954.81005859, 1960.03503418,
                                    1971.47998047, 1976.70495605, 1988.15002441, 1993.375     ,
                                    2004.81994629, 2010.04504395])

X = np.load('/beegfs/desy/user/akorol/data/calo-clouds/muon-map/X.npy')
Z = np.load('/beegfs/desy/user/akorol/data/calo-clouds/muon-map/Z.npy')
Y = np.load('/beegfs/desy/user/akorol/data/calo-clouds/muon-map/Y.npy')
E = np.load('/beegfs/desy/user/akorol/data/calo-clouds/muon-map/E.npy')

inbox_idx = np.where((Y > Ymin) & (X < Xmax) & (X > Xmin) & (Z < Zmax) & (Z > Zmin) )[0]


X = X[inbox_idx]
Z = Z[inbox_idx]
Y = Y[inbox_idx]
E = E[inbox_idx]


def create_map(X, Y, Z, dm=3):
    """
        X, Y, Z: np.array 
            ILD coordinates of sencors hited with muons
        dm: int (1, 2, 3, 4, 5) dimension split multiplicity
    """

    offset = half_cell_size*2/(dm)

    layers = []
    for l in tqdm(range(len(layer_bottom_pos))): # loop over layers
        idx = np.where((Y <= (layer_bottom_pos[l] + cell_thickness*1.5)) & (Y >= layer_bottom_pos[l] - cell_thickness/2 ))
        
        xedges = np.array([])
        zedges = np.array([])
        
        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])
        
        xedges = np.append(xedges, unique_X[0] - half_cell_size)
        xedges = np.append(xedges, unique_X[0] + half_cell_size)
        
        for i in range(len(unique_X)-1): # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i+1]) > half_cell_size * 1.9:
                xedges = np.append(xedges, unique_X[i+1] - half_cell_size)
                xedges = np.append(xedges, unique_X[i+1] + half_cell_size)
                
                for of_m in range(dm):
                    xedges = np.append(xedges, unique_X[i+1] - half_cell_size + offset*of_m) # for higher granularity
                
        for z in unique_Z: # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size)
            zedges = np.append(zedges, z + half_cell_size)
            
            for of_m in range(dm):
                zedges = np.append(zedges, z - half_cell_size + offset*of_m) # for higher granularity
                
            
        zedges = np.unique(zedges)
        xedges = np.unique(xedges)
        
        xedges = [xedges[i] for i in range(len(xedges)-1) if abs(xedges[i] - xedges[i+1]) > 1e-3] + [xedges[-1]]
        zedges = [zedges[i] for i in range(len(zedges)-1) if abs(zedges[i] - zedges[i+1]) > 1e-3] + [zedges[-1]]
        
            
        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({'xedges': xedges, 'zedges': zedges, 'grid': H})

    return layers, offset




def get_projections(showers, MAP, layer_bottom_pos, return_cell_point_cloud=False):
    events = []
    
    for shower in tqdm(showers):
        layers = []
        
        x_coord, y_coord, z_coord, e_coord = shower

        
        for l in range(len(MAP)):
            idx = np.where((y_coord <= (layer_bottom_pos[l] + 1)) & (y_coord >= layer_bottom_pos[l] - 0.5 ))
            
            xedges = MAP[l]['xedges']
            zedges = MAP[l]['zedges']
            H_base = MAP[l]['grid']
            
            H, xedges, zedges = np.histogram2d(x_coord[idx], z_coord[idx], bins=(xedges, zedges), weights=e_coord[idx])
            H[H_base==0] = 0
            
            layers.append(H)
        
        events.append(layers)
    
    if not return_cell_point_cloud:
        return events
    
    else:
        pass




def get_cog(x, y, z, e):
    x_cog = np.sum((x * e), axis=1) / e.sum(axis=1)
    y_cog = np.sum((y * e), axis=1) / e.sum(axis=1)
    z_cog = np.sum((z * e), axis=1) / e.sum(axis=1)
    return x_cog, y_cog, z_cog

def get_features(events, thr=0.05):
    
    incident_point = (0, 40)
    
    occ_list = [] # occupancy
    hits_list = [] # energy per cell
    e_sum_list = [] # energy per shower
    e_radial = [] # radial profile
    e_layers_list = [] # energy per layer

    for layers in tqdm(events):

        occ = 0
        e_sum = 0
        e_layers = []
        y_pos = []
        for l, layer in enumerate(layers):
            layer = layer*1000 # energy rescale
            layer[layer < thr] = 0

            hit_mask = layer > 0
            layer_hits = layer[hit_mask]
            layer_sum = layer.sum()

            occ += hit_mask.sum()
            e_sum += layer.sum()

            hits_list.append(layer_hits)
            e_layers.append(layer.sum())


            # get radial profile #######################
            x_hit_idx, z_hit_idx = np.where(hit_mask)
            x_cell_coord = MAP[l]['xedges'][:-1][x_hit_idx] + half_cell_size
            z_cell_coord = MAP[l]['zedges'][:-1][z_hit_idx] + half_cell_size
            e_cell = layer[x_hit_idx, z_hit_idx]
            dist_to_origin = np.sqrt((x_cell_coord - incident_point[0])**2 + (z_cell_coord - incident_point[1])**2)
            e_radial.append([dist_to_origin, e_cell])
            ############################################




        e_layers_list.append(e_layers)

        occ_list.append(occ)
        e_sum_list.append(e_sum)

    e_radial = np.concatenate(e_radial, axis=1)
    occ_list = np.array(occ_list)
    e_sum_list = np.array(e_sum_list)
    hits_list = np.concatenate(hits_list)
    e_layers_list = np.array(e_layers_list).sum(axis=0)/len(events)
    
    return e_radial, occ_list, e_sum_list, hits_list, e_layers_list


def plt_radial(e_radial, e_radial_list, labels, cfg=cfg):
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)
    h = ax1.hist(e_radial[0], bins=cfg.bins_r, weights=e_radial[1], color='lightgrey', label=labels[0])
    h = ax1.hist(e_radial[0], bins=cfg.bins_r, weights=e_radial[1], color='dimgrey', histtype='step', lw=1.5)
    
    for i, e_radial_ in enumerate(e_radial_list):
        h = ax1.hist(e_radial_[0], bins=h[1], weights=e_radial_[1], histtype='step', lw=2, label=labels[i+1])
    
    
    ax1.set_yscale('log')

    ax2 = ax1.twiny()
    ax2.set_xticks( ax1.get_xticks() )
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels([int(x / (half_cell_size*2)) for x in ax1.get_xticks()])

    ax2.set_xlabel("radius [cells]", fontsize=15, family='serif')
    ax1.set_xlabel("radius [mm]", fontsize=15, family='serif')
    ax1.set_ylabel('energy sum [MeV]', fontsize=15, family='serif')
    
    ax1.legend(prop=cfg.font, loc='upper right')

    plt.tight_layout()

    plt.savefig('radial.png', dpi=300)
    plt.show()
    
def plt_spinal(e_layers, e_layers_list, labels, cfg=cfg):
    
    plt.figure(figsize=(7,7))
    plt.hist(np.arange(len(e_layers)), bins=30, weights=e_layers, color='lightgrey', label=labels[0])
    plt.hist(np.arange(len(e_layers)), bins=30, weights=e_layers, color='dimgrey', histtype='step', lw=1.5)
    
    for i, e_layers_ in enumerate(e_layers_list):
        plt.hist(np.arange(len(e_layers_)), bins=30, weights=e_layers_, histtype='step', lw=2, label=labels[i+1])

    plt.yscale('log')
    plt.ylim(1, 1000)
    plt.xlabel('layers', fontsize=15, family='serif')
    plt.ylabel('energy sum [MeV]', fontsize=15, family='serif')
    
    plt.legend(prop=cfg.font, loc='upper left')
    plt.tight_layout()

    plt.savefig('spinal.png', dpi=300)
    plt.show()
    
def plot_occupancy(occ, occ_list, labels, cfg=cfg):
    plt.figure(figsize=(7,7))
    h = plt.hist(occ, bins=cfg.occup_bins, label=labels[0], color='lightgrey')
    h = plt.hist(occ, bins=cfg.occup_bins, color='dimgrey', histtype='step')
    
    for i, occ_ in enumerate(occ_list):
        plt.hist(occ_, bins=h[1], histtype='step', lw=2, label=labels[i+1], ls='-')

    # plt.xlim(0, 1500)
    plt.xlabel('Num. hits', fontsize=15, family='serif')
    plt.ylabel('Counts', fontsize=15, family='serif')
    plt.legend(prop=cfg.font, loc='upper left')
    plt.tight_layout()
    plt.savefig('occ.png', dpi=200)
    plt.show()
    
def plt_hit_e(hits, hits_list, labels, cfg=cfg):
    plt.figure(figsize=(7,7))
    h = plt.hist(hits, bins=cfg.hit_bins, label=labels[0], color='lightgrey')
    h = plt.hist(hits, bins=cfg.hit_bins, histtype='step', color='dimgrey')
    
    for i, hits_ in enumerate(hits_list):
        plt.hist(hits_, bins=h[1], histtype='step', lw=2, label=labels[i+1], ls='-')

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Cell Energy [MeV]', fontsize=15, family='serif')
    plt.ylabel('Counts', fontsize=15, family='serif')
    plt.legend(prop=cfg.font, loc='lower left')
    plt.tight_layout()
    plt.savefig('hits.png', dpi=200)
    plt.show()
    
def plt_esum(e_sum, e_sum_list, labels, cfg=cfg):
    plt.figure(figsize=(7, 7))
    h = plt.hist(np.array(e_sum), bins=40, label=labels[0], color='lightgrey')
    
    for i, e_sum_ in enumerate(e_sum_list):
        plt.hist(np.array(e_sum_), bins=h[1], histtype='step', lw=2, label=labels[i+1])

    plt.xlim(0, 2500)
    plt.xlabel('Energy Sum [MeV]', fontsize=15, family='serif')
    plt.ylabel('Counts', fontsize=15, family='serif')
    plt.legend(prop=cfg.font, loc='upper left')
    plt.tight_layout()
    plt.savefig('e_sum.png', dpi=200)
    plt.show()

def plt_cog(cog, cog_list, labels, cfg=cfg):
    lables = ["x", "y", "z"]
    plt.figure(figsize=(7,21))

    for j in range(3):
        plt.subplot(3, 1, j+1)
        
        
        if j == 0:
            h = plt.hist(np.array(cog[j]), bins=cfg.bins_cog, label=labels[0], color='lightgrey', range=(-4, 4))
            plt.xlim(-4, 4)
        elif j == 2:
            h = plt.hist(np.array(cog[j]), bins=cfg.bins_cog, label=labels[0], color='lightgrey', range=(36, 44))
            plt.xlim(36, 44)
        else:
            h = plt.hist(np.array(cog[j]), bins=cfg.bins_cog, label=labels[0], color='lightgrey')

        for i, cog_ in enumerate(cog_list):

            if j == 0:
                h = plt.hist(np.array(cog_[j]), bins=h[1], histtype='step', lw=2, label=labels[i+1], range=(-4, 4))
                plt.xlim(-4, 4)
            elif j == 2:
                h = plt.hist(np.array(cog_[j]), bins=h[1], histtype='step', lw=2, label=labels[i+1], range=(36, 44))
                plt.xlim(36, 44)
            else:
                plt.hist(np.array(cog_[j]), bins=h[1], histtype='step', lw=2, label=labels[i+1])

        if j == 1:
            plt.legend(prop=cfg.font)

        plt.xlabel(f'CoG {lables[j]} [mm]', fontsize=15, family='serif')
        plt.ylabel('Counts', fontsize=15, family='serif')

    
    plt.tight_layout()
    plt.savefig('cog.png', dpi=200)
    plt.show()



def get_plots(events, events_list: list, labels: list = ['1', '2', '3'], thr=0.05):
    
    e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real = get_features(events, thr)
    
    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = [], [], [], [], []
    
    for i in range(len(events_list)):
        e_radial_, occ_real_, e_sum_real_, hits_real_, e_layers_real_ = get_features(events_list[i])
        
        e_radial_list.append(e_radial_)
        occ_list.append(occ_real_)
        e_sum_list.append(e_sum_real_)
        hits_list.append(hits_real_)
        e_layers_list.append(e_layers_real_)
        
    
    plt_radial(e_radial_real, e_radial_list, labels=labels)
    plt_spinal(e_layers_real, e_layers_list, labels=labels)
    plot_occupancy(occ_real, occ_list, labels=labels)
    plt_hit_e(hits_real, hits_list, labels=labels)
    plt_esum(e_sum_real, e_sum_list, labels=labels)


MAP, offset = create_map(X, Y, Z, dm=1)
