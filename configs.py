class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = 'point-cloud_v1'

    # Model arguments
        self.model = 'flow'             # choices=['flow', 'gaussian']
        self.latent_dim = 69
        self.num_steps = 68
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.sched_mode = 'linear'
        self.flexibility = 0.0
        self.truncate_std = 2.0
        self.latent_flow_depth = 13
        self.latent_flow_hidden_dim = 126
        self.num_samples = 4
        self.sample_num_points = 2048
        self.kl_weight = 0.001698
        self.residual = True            # choices=[True, False]
        self.spectral_norm = False      # choices=[True, False]


    # Data
        self.dataset_path = '/beegfs/desy/user/akorol/projects/getting_high/ILDConfig/StandardConfig/production/out/no_projection_clastered_10-90GV_115k_sorted.hdf5'
    # Dtataloader
        self.workers = 64
        self.train_bs = 128
        self.pin_memory = False         # choices=[True, False]
        self.shuffle = True             # choices=[True, False]

    # Optimizer and scheduler
        self.lr = 2e-3
        self.weight_decay = 0
        self.max_grad_norm = 10
        self.end_lr = 1e-4
        self.sched_start_epoch = 100 * 1e3
        self.sched_end_epoch = 400 * 1e3

    # Others
        self.device = 'cuda'
        self.logdir = '/beegfs/desy/user/akorol/logs/point-cloud'
        self.seed = 42
        self.max_iters = 300 * 1e3
        self.val_freq = 1e3
        self.test_freq = 30 * 1e3
        self.test_size = 400
        self.tag = None
