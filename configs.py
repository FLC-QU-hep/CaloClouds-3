class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = 'CD_'  # options: [TEST_, kCaloClouds_, CaloClouds_, CD_]
        self.comet_project = 'calo-consistency'   # options: ['k-CaloClouds', 'calo-consistency']
        #self.Acomment = 'baseline with lat_dim = 32, max_iter 10M, lr=1e-4 FIXED, dropout_rate=0.0, ema_power=2/3 (long training)'  # log_iter 100
        self.Acomment = 'long baseline with lat_dim = 32, max_iter 10M, lr=2e-4 fixed, num_steps=18, bs=64, simga_max=80, epoch=2M, EMA'  # log_iter 100
        self.log_comet = True

    # Model arguments
        self.model_name = 'epicVAE_nFlow_kDiffusion'             # choices=['flow', 'AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion]
        self.latent_dim = 32     # caloclouds default: 256
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
        self.kl_weight = 1e-3   # default: 0.001 = 1e-3
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

    # Dataloader
        self.workers = 32
        self.train_bs = 64      # k-diffusion: 128 / CD: 256
        self.pin_memory = False         # choices=[True, False]
        self.shuffle = True             # choices=[True, False]
        self.max_points = 6_000
        

    # Optimizer and scheduler
        self.optimizer = 'RAdam'         # choices=['Adam', 'RAdam']
        self.lr = 2e-4              # Caloclouds default: 2e-3, consistency model paper: approx. 1e-5
        self.end_lr = 2e-4
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.sched_start_epoch = 100 * 1e3
        self.sched_end_epoch = 400 * 1e3
        self.max_iters = 10 * 1e6

    # Others
        self.device = 'cuda'
        self.logdir = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/log'
        self.seed = 42
        self.val_freq =  10_000  #  1e3          # saving intervall for checkpoints

        self.test_freq = 30 * 1e3   
        self.test_size = 400
        self.tag = None
        self.log_iter = 100   # log every n iterations, default: 100

    # EMA scheduler
        self.ema_type = 'inverse'
        self.ema_power = 0.6667   # depends on the number of iterations, 2/3=0.6667 good for 1e6 iterations, 3/4=0.75 good for less
        self.ema_max_value = 0.9999
        
    # EDM diffusion parameters for training
        self.model = {
            # "sigma_data" : [0.08, 0.35, 0.08, 0.5],    ## default parameters for EDM pape = 0.5, might need to adjust for our dataset (meaning the std of our data) / or a seperate sigma for each feature?
            "sigma_data" : 0.5,
            # "has_variance" : False,
            # "loss_config" : "karras",
            "sigma_sample_density" : {
                "type": "lognormal",
                "mean": -1.2,
                "std": 1.2
                }
            }
        self.dropout_mode = 'all'     # options: 'all',  'mid'  location of the droput layers
        self.dropout_rate = 0.0       # EDM: approx. 0.1, Caloclouds default: 0.0
        self.diffusion_loss = 'l2'    # l2 or l1

    # EDM diffusion parameters for sampling    / also used in CM distillation
        self.num_steps = 18      # EDM paper: 18
        self.sampler = 'heun'
        self.sigma_min = 0.002  # EDM paper: 0.002, k-diffusion config: 0.01
        self.sigma_max = 80.0
        self.rho = 7.0    # exponent in EDM boundaries
        self.s_churn = 0.0
        self.s_noise = 1.0


    # Consistency Distillation parameters
        # self.model_path = 'kCaloClouds_2023_05_24__14_54_09/ckpt_0.000000_500000.pt'
        # self.model_path = 'kCaloClouds_2023_05_31__17_57_11/ckpt_0.000000_1690000.pt'
        # self.model_path = 'kCaloClouds_2023_06_29__23_08_31/ckpt_0.000000_2000000.pt'   # lat 0
        self.model_path = 'kCaloClouds_2023_07_02__20_30_03/ckpt_0.000000_2000000.pt'     # lat 32
        self.use_ema_trainer = True
        self.start_ema = 0.95
        # self.ema_rate = 0.999943    # decay rate of separately saved EMA model (not implemented yet)
        self.cm_random_init = False    # kinda like consistency training, but still with a teacher score function

    