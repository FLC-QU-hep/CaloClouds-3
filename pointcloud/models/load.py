from pointcloud.models.diffusion import Diffusion


def get_model_class(config):
    if config.model_name == "Diffusion":
        m_class = Diffusion
    else:
        raise NotImplementedError(
            f"Model {config.model_name} not implemented, known models: "
            "AllCond_epicVAE_nFlow_PointDiff, Diffusion, wish, fish"
        )
    return m_class
