class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = 'TEST_'  
        self.Acomment = 'running baselione CaloClouds with default parameters'
        self.comet_project = 'k-CaloClouds'    # project name in comet.ml
        self.log_comet = False

    # Model arguments
        self.model = 'AllCond_epicVAE_nFlow_PointDiff'             # choices=['flow', 'AllCond_epicVAE_nFlow_PointDiff']
        self.latent_dim = 256
        self.num_steps = 100
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.sched_mode = 'quardatic'  # options: ['linear', 'quardatic', 'sigmoid]
        self.flexibility = 0.0
        self.truncate_std = 2.0
        self.latent_flow_depth = 14
        self.latent_flow_hidden_dim = 256
        self.num_samples = 4
        self.features = 4
        self.sample_num_points = 2048
        self.kl_weight = 0.001
        self.residual = True            # choices=[True, False]
        self.spectral_norm = False      # choices=[True, False]
        
        self.cond_features = 2       # number of conditioning features (i.e. energy+points=2)
        self.norm_cond = True    # normalize conditioniong to [-1,1]
        self.kld_min = 1.0       # default: 0.0

        # EPiC arguments
        self.use_epic = False
        self.epic_layers = 5
        self.hid_d = 128        # default: 128
        self.sum_scale = 1e-3
        self.weight_norm = True

        # for n_flows model
        self.flow_model = 'PiecewiseRationalQuadraticCouplingTransform'
        self.flow_transforms = 10
        self.flow_layers = 2
        self.flow_hidden_dims = 128
        self.tails = 'linear'
        self.tail_bound = 10
        


    # Data
        self.dataset = 'x36_grid' # choices=['x36_grid', 'clustered', 'getting_high']
        # self.dataset_path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/clustered/10-90GeV_clustered_524k_sorted_float32.hdf5'
        # self.dataset_path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k.hdf5'
        self.dataset_path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'
        # self.dataset_path = '/beegfs/desy/user/korcariw/CaloClouds/dataset/showers/photons_10_100GeV_float32_sorted_train.h5'
        # self.dataset_path = '/beegfs/desy/user/korcariw/CaloClouds/dataset/showers/photons_50GeV_sorted.h5'
        # self.dataset_path = '/beegfs/desy/user/akorol/projects/getting_high/ILDConfig/StandardConfig/production/out/photons_50GeV_40k.slcio.hdf5'
        self.quantized_pos = False

    # Dtataloader
        self.workers = 32
        self.train_bs = 256
        self.pin_memory = False         # choices=[True, False]
        self.shuffle = True             # choices=[True, False]
        self.max_points = 6_000
        

    # Optimizer and scheduler
        self.lr = 2e-3
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.end_lr = 1e-4
        self.sched_start_epoch = 300 * 1e3
        self.sched_end_epoch = 2 * 1e6

    # Others
        self.device = 'cuda'
        self.logdir = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/log'
        self.seed = 42
        self.max_iters = 150 # 2 * 1e6
        self.val_freq =  100  #  1e3          # saving intervall for checkpoints
        self.test_freq = 30 * 1e3
        self.test_size = 400
        self.tag = None
        self.log_iter = 10   # log every n iterations

    # EMA scheduler
        self.ema_type = 'inverse'
        self.ema_power = 0.6667   # depends on the number of iterations, 2/3 good for 1e6 iterations, 3/4 good for less
        self.ema_max_value = 0.9999
        

    