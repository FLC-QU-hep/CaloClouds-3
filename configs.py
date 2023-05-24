class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = 'kCaloClouds_'  # options: [TEST_, kCaloClouds_, CaloClouds_]
        self.Acomment = 'first run with EDM CaloClouds setup, 500k iterations, scheduler from 100k-400k, EMApower 0.75'
        self.comet_project = 'k-CaloClouds'    # project name in comet.ml
        self.log_comet = True

    # Model arguments
        self.model_name = 'epicVAE_nFlow_kDiffusion'             # choices=['flow', 'AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion]
        self.latent_dim = 256
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
        self.residual = False            # choices=[True, False]   # !! for CaloClouds was True, but for EDM False might be better (?)
        
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
        self.train_bs = 128
        self.pin_memory = False         # choices=[True, False]
        self.shuffle = True             # choices=[True, False]
        self.max_points = 6_000
        

    # Optimizer and scheduler
        self.lr = 2e-3
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.end_lr = 1e-4
        self.sched_start_epoch = 100 * 1e3
        self.sched_end_epoch = 400 * 1e3

    # Others
        self.device = 'cuda'
        self.logdir = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/log'
        self.seed = 42
        self.max_iters = 500_000 # 2 * 1e6
        self.val_freq =  1000  #  1e3          # saving intervall for checkpoints

        self.test_freq = 30 * 1e3
        self.test_size = 400
        self.tag = None
        self.log_iter = 100   # log every n iterations

    # EMA scheduler
        self.ema_type = 'inverse'
        self.ema_power = 0.75   # depends on the number of iterations, 2/3=0.6667 good for 1e6 iterations, 3/4=0.75 good for less
        self.ema_max_value = 0.9999
        
    # EDM diffusion parameters for training
        self.model = {
            "sigma_data" : 0.5,    ## default parameters for EDM paper, might need to adjust for our dataset (meaning the std of our data) / or a seperate sigma for each feature?
            "has_variance" : False,
            "loss_config" : "karras",
            "sigma_sample_density" : {
                "type": "lognormal",
                "mean": -1.2,
                "std": 1.2
                }
            }

    # EDM diffusion parameters for sampling
        self.num_steps = 100
        self.sigma_min = 0.002  # EDM paper: 0.002, k-diffusion config: 0.01
        self.sigma_max = 80.0


    