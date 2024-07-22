from pointcloud.models.vae_flow import VAEFlow
from pointcloud.models.allCond_epicVAE_nflow_PointDiff import (
    AllCond_epicVAE_nFlow_PointDiff,
)
from pointcloud.models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
from pointcloud.models.wish import Wish


def get_model_class(configs):
    if configs.model_name == "flow":
        m_class = VAEFlow
    elif configs.model_name == "AllCond_epicVAE_nFlow_PointDiff":
        m_class = AllCond_epicVAE_nFlow_PointDiff
    elif configs.model_name == "epicVAE_nFlow_kDiffusion":
        m_class = epicVAE_nFlow_kDiffusion
    elif configs.model_name == "wish":
        m_class = Wish
    else:
        raise NotImplementedError(
            f"Model {configs.model_name} not implemented, known models: "
            "flow, AllCond_epicVAE_nFlow_PointDiff, epicVAE_nFlow_kDiffusion, wish"
        )
    return m_class
