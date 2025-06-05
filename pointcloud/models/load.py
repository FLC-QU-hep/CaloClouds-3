from pointcloud.models.allCond_epicVAE_nflow_PointDiff import (
    AllCond_epicVAE_nFlow_PointDiff,
)
from pointcloud.models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
from pointcloud.models.wish import Wish
from pointcloud.models.fish import Fish


def get_model_class(config):
    if config.model_name == "AllCond_epicVAE_nFlow_PointDiff":
        m_class = AllCond_epicVAE_nFlow_PointDiff
    elif config.model_name == "epicVAE_nFlow_kDiffusion":
        m_class = epicVAE_nFlow_kDiffusion
    elif config.model_name == "wish":
        m_class = Wish
    elif config.model_name == "fish":
        m_class = Fish
    else:
        raise NotImplementedError(
            f"Model {config.model_name} not implemented, known models: "
            "AllCond_epicVAE_nFlow_PointDiff, epicVAE_nFlow_kDiffusion, wish, fish"
        )
    return m_class
