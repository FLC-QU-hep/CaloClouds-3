"""
Get/set the precision for various parts of the dataset
"""
import torch

default_component_precisions = {
    "anomaly_tree_feat": torch.float32,
    "diffusion": torch.float32,
    "showerflow": torch.float32,
    "wish": torch.float32,
    "fish": torch.float32,
}


def get(component_name, settings=None):
    """
    Check what precision is required for a specified component of the model

    Parameters
    ----------
    component_name : str
        The name of the component to check precision for.
    settings : pointcloud.configs.Configs (optional)
        A Configs object from which precisions can be extracted,
        the function expects the configs object to have an attribute
        with name '{component}_precision' which should contain the
        precision to be used for the specified component.
        Otherwise, the default precision for this component will be returned.
    Returns
    -------
    precision : torch.dtype
        The precision required for the specified component.
    """
    if settings is not None:
        precision = getattr(settings, f"{component_name}_precision", None)
        if precision is not None:
            if isinstance(precision, str):
                precision = getattr(torch, precision)
            return precision
    return default_component_precisions[component_name]


def set(component, component_name, setting=None):
    """
    Set the precision for a specified component of the model.

    Parameters
    ----------
    component : tensor or other
        The component to set the precision for.
    component_name : str
        The name of the component to set the precision for.
    settings : pointcloud.configs.Configs (optional)
        A Configs object from which precisions can be extracted,
        the function expects the configs object to have an attribute
        with name 'precision_{component}' which should contain the
        precision to be used for the specified component.
        Otherwise, the default precision for this component will be returned.
    Returns
    -------
    component : tensor or other
        The component with the precision set. May be self or a new object.
    """
    torch_tensor = isinstance(component, torch.Tensor)
    if torch_tensor:
        component = component.to(get(component_name, setting))
    else:
        raise NotImplementedError(
            f"Setting precision for {type(component)} is not yet implemented."
        )
    return component
