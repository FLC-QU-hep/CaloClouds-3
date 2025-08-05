import os
import torch
import numpy as np
import random
import time
from nflows import transforms, distributions, flows
import torch.nn as nn
from typing import Callable
from ..data.conditioning import get_cond_dim

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class CheckpointManager(object):
    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != "ckpt":
                continue
            _, score, it = f.split("_")
            it = it.split(".")[0]
            self.ckpts.append(
                {
                    "score": float(score),
                    "file": f,
                    "iteration": int(it),
                }
            )

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float("-inf")
        for i, ckpt in enumerate(self.ckpts):
            if ckpt["score"] >= worst:
                idx = i
                worst = ckpt["score"]
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float("inf")
        for i, ckpt in enumerate(self.ckpts):
            if ckpt["score"] <= best:
                idx = i
                best = ckpt["score"]
        return idx if idx >= 0 else None

    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        try:
            for i, ckpt in enumerate(self.ckpts):
                if ckpt["iteration"] > latest_it:
                    idx = i
                    latest_it = ckpt["iteration"]
        except KeyError:
            print(
                "No iteration information found in checkpoint filenames."
                "Returning the latest checkpoint based on the file timestamp."
            )
            timestamps = [
                os.path.getmtime(os.path.join(self.save_dir, ckpt["file"]))
                for ckpt in self.ckpts
            ]
            print(timestamps)
            if timestamps:
                idx = np.argmax(timestamps)
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):
        ckpt = {"score": score}
        score_part = f"{float(score):.6f}" if np.isfinite(score) else "unscored"
        if step is None:
            step_part = "None"
        else:
            step_part = f"{int(step)}"
            ckpt["iteration"] = step
        fname = f"ckpt_{score_part}_{step_part}.pt"
        path = os.path.join(self.save_dir, fname)
        ckpt["file"] = fname

        path = os.path.join(self.save_dir, fname)
        torch.save(
            {"args": args, "state_dict": model.state_dict(), "others": others}, path
        )

        self.ckpts.append(ckpt)
        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError("No checkpoints found.")
        ckpt = torch.load(
            os.path.join(self.save_dir, self.ckpts[idx]["file"]), weights_only=False
        )
        return ckpt

    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError("No checkpoints found.")
        ckpt = torch.load(
            os.path.join(self.save_dir, self.ckpts[idx]["file"]), weights_only=False
        )
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file), weights_only=False)
        return ckpt


def seed_all(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def get_new_log_dir(root="./logs", postfix="", prefix="", start_time=time.localtime()):
    if not os.path.exists(root):
        print(f"Also creating directory {root}")
    else:
        assert os.path.isdir(root), f"{root} is not a directory, cant create log dir"
    log_dir = os.path.join(
        root, prefix + time.strftime("%Y_%m_%d__%H_%M_%S", start_time) + postfix
    )
    if os.path.exists(log_dir):
        print(f"Directory {log_dir} already exists, trying another one.")
        postfix += "_{}".format(random.randint(0, 1000))
        log_dir = get_new_log_dir(root, postfix, prefix, start_time)
        return log_dir
    os.makedirs(log_dir)
    return log_dir


def get_flow_model(cfg):
    if cfg.flow_model == "MaskedPiecewiseRationalQuadraticAutoregressiveTransform":
        flow = get_maf_spline_flow(cfg)
    elif cfg.flow_model == "PiecewiseRationalQuadraticCouplingTransform":
        flow = get_coupling_spline_flow(cfg)
    else:
        raise NotImplementedError("flow model not defined")
    return flow


def get_maf_spline_flow(cfg):
    params_flow = {
        "features": cfg.latent_dim,
        "context_features": get_cond_dim(cfg, "diffusion"),
        "hidden_features": cfg.flow_hidden_dims,
        "num_blocks": cfg.flow_layers,
        "activation": nn.LeakyReLU(),
        "use_residual_blocks": True,
        "tails": cfg.tails,
        "tail_bound": cfg.tail_bound,
    }
    # Define an invertible transformation.
    t_list = []
    for _ in range(cfg.flow_transforms):
        t_list.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **params_flow
            )
        )
        t_list.append(transforms.RandomPermutation(features=cfg.latent_dim))
    transform = transforms.CompositeTransform(t_list)
    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[cfg.latent_dim])
    # Combine into a flow.
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    return flow


def get_coupling_spline_flow(cfg):
    params_subnet = {
        "hidden_features": cfg.flow_hidden_dims,
        "context_features": get_cond_dim(cfg, "diffusion"),
        "num_layers": cfg.flow_layers,
        "activation": nn.LeakyReLU,
        "dropout_probability": 0.0,
    }
    subnet = SubnetFactory(**params_subnet)
    params_flow = {
        "tails": cfg.tails,
        "tail_bound": cfg.tail_bound,
        "transform_net_create_fn": subnet,
        "mask": torch.arange(0, cfg.latent_dim) < cfg.latent_dim // 2,
    }
    # Define an invertible transformation.
    t_list = []
    for _ in range(cfg.flow_transforms):
        t_list.append(
            transforms.PiecewiseRationalQuadraticCouplingTransform(**params_flow)
        )
        t_list.append(transforms.RandomPermutation(features=cfg.latent_dim))
    transform = transforms.CompositeTransform(t_list)
    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[cfg.latent_dim])
    # Combine into a flow.
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    return flow


class SubnetFactory:
    class CatCall(nn.Module):
        def __init__(self, layers) -> None:
            super().__init__()
            self.layers = layers

        def forward(self, x, context):
            if context is not None:
                x = torch.cat((x, context), dim=1)
            return self.layers(x)

    def __init__(
        self,
        hidden_features: int,
        context_features: int,
        num_layers: int,
        activation: Callable,
        dropout_probability: float,
    ) -> None:
        self.context_features = context_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_probability = dropout_probability

    def __call__(self, num_features_in, num_features_out):
        layers = []
        last_features = num_features_in + self.context_features
        for i in range(self.num_layers):
            layers.append(nn.Linear(last_features, self.hidden_features))
            layers.append(self.activation())
            if self.dropout_probability > 0.0:
                layers.append(nn.Dropout(self.dropout_probability))
            last_features = self.hidden_features
        layers.append(nn.Linear(last_features, num_features_out))
        return self.CatCall(nn.Sequential(*layers))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
