from . import default
import os


class Configs(default.Configs):
    def __init__(self):
        self.latent_dim = 0  # no latent flow in new calocloud
        
