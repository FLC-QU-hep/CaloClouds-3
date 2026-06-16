from pointcloud.config_varients import caloclouds_3


class Configs(caloclouds_3.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_kwargs(kwargs)
