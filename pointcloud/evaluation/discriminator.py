"""
Using a simple feedforward fully connected network, evaluate models.
Takes high level features.
"""

import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from ..data import read_write, naming
from ..data.conditioning import read_raw_regaxes_withcond, normalise_cond_feats
from ..utils import showerflow_training, showerflow_utils, gen_utils
from ..utils.metadata import Metadata
from ..models import shower_flow
from . import generate
import collections


# some globals that are growing...
descriminator_params = {
    "settings0": {
        "n_input_features": 63,
        "events_per_input": 10,
        "n_hidden_layers": 5,
        "n_hidden_units": 20,
    },
    "settings1": {
        "n_input_features": 63,
        "events_per_input": 1,
        "n_hidden_layers": 2,
        "n_hidden_units": 10,
    },
    "settings2": {
        "n_input_features": 63,
        "events_per_input": 1,
        "n_hidden_layers": 2,
        "n_hidden_units": 4,
    },
    "settings3": {
        "n_input_features": 62,
        "events_per_input": 1,
        "n_hidden_layers": 2,
        "n_hidden_units": 4,
    },
    "settings4": {
        "n_input_features": 3,
        "events_per_input": 1,
        "n_hidden_layers": 2,
        "n_hidden_units": 4,
    },
    "settings5": {
        "n_input_features": 3,
        "events_per_input": 1,
        "n_hidden_layers": 5,
        "n_hidden_units": 10,
    },
    "settings6": {
        "n_input_features": 30,
        "events_per_input": 1,
        "n_hidden_layers": 5,
        "n_hidden_units": 10,
    },
    "settings7": {
        "n_input_features": 30,
        "events_per_input": 1,
        "n_hidden_layers": 5,
        "n_hidden_units": 10,
    },
    "settings8": {
        "n_input_features": 2,
        "events_per_input": 1,
        "n_hidden_layers": 5,
        "n_hidden_units": 10,
    },
    "settings9": {
        "n_input_features": 62,
        "events_per_input": 1,
        "n_hidden_layers": 5,
        "n_hidden_units": 10,
    },
    "settings10": {
        "n_input_features": 62,
        "events_per_input": 10,
        "n_hidden_layers": 5,
        "n_hidden_units": 10,
    },
    "settings11": {
        "n_input_features": 62,
        "events_per_input": 10,
        "n_hidden_layers": 10,
        "n_hidden_units": 20,
    },
    "settings12": {
        "n_input_features": 62,
        "events_per_input": 1,
        "n_hidden_layers": 10,
        "n_hidden_units": 20,
    },
}

mask3 = np.ones(63, dtype=bool)
mask3[2] = False
mask4 = np.zeros(63, dtype=bool)
mask4[:3] = True
mask6 = np.zeros(63, dtype=bool)
mask6[3:33] = True
mask7 = np.zeros(63, dtype=bool)
mask7[33:] = True
mask8 = np.zeros(63, dtype=bool)
mask8[[18, 48]] = True

feature_masks = {
    "settings0": None,
    "settings1": None,
    "settings2": None,
    "settings3": mask3,
    "settings4": mask4,
    "settings5": mask4,
    "settings6": mask6,
    "settings7": mask7,
    "settings8": mask8,
    "settings9": mask3,
    "settings10": mask3,
    "settings11": mask3,
    "settings12": mask3,
}


class Discriminator(nn.Module):
    def __init__(
        self, n_input_features, events_per_input, n_hidden_layers, n_hidden_units
    ):
        super().__init__()
        self.n_input_features = n_input_features
        self.events_per_input = events_per_input
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.layers = []
        self.layers.append(
            nn.Linear(
                self.n_input_features * self.events_per_input, self.n_hidden_units
            )
        )
        self.layers.append(nn.ReLU())
        for h in range(self.n_hidden_layers - 1):
            self.layers.append(nn.Linear(self.n_hidden_units, self.n_hidden_units))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.n_hidden_units, 1))
        self.layers.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*self.layers)
        self._parameters = collections.OrderedDict(self.sequential.named_parameters())

    def forward(self, x):
        return self.sequential(x)

    def save(self, file_path):
        init_args = {
            "n_input_features": self.n_input_features,
            "events_per_input": self.events_per_input,
            "n_hidden_layers": self.n_hidden_layers,
            "n_hidden_units": self.n_hidden_units,
        }
        to_save = {
            "init_args": init_args,
            "state_dict": self.sequential.state_dict(),
        }
        torch.save(to_save, file_path)

    @classmethod
    def load(cls, file_path):
        loaded = torch.load(file_path, weights_only=False)
        model = cls(**loaded["init_args"])
        model.sequential.load_state_dict(loaded["state_dict"])
        return model

    def parameters(self):
        return self._parameters


class Features:
    def __init__(self, feature_mask=None):
        if feature_mask is None:
            self.mask = slice(None)
        else:
            self.mask = feature_mask
        self._unmasked_n_features = None
        self._n_features = None
        self.file_names = []
        self.file_lengths = np.array([], dtype=int)
        self.first_idxs = np.array([0], dtype=int)
        self._loaded_file = None
        self._loaded_file_idx = None

    def add_file(self, file_path):
        data = np.load(file_path)
        if self._unmasked_n_features is None:
            self._unmasked_n_features = data.shape[1]
            self._n_features = len(np.empty(self._unmasked_n_features)[self.mask])
        else:
            assert (
                data.shape[1] == self._unmasked_n_features
            ), "Number of features in file does not match previous files"
        self.file_names.append(file_path)
        self.file_lengths = np.append(self.file_lengths, data.shape[0])
        self.first_idxs = np.append(
            self.first_idxs, self.first_idxs[-1] + self.file_lengths[-1]
        )
        del data

    def __len__(self):
        return self.first_idxs[-1]

    def _idx_in_file(self, idx):
        file_idx = np.searchsorted(self.first_idxs, idx, side="right") - 1
        interal_idx = idx - self.first_idxs[file_idx]
        return file_idx, interal_idx

    def _get_from_file(self, file_idx, internal_idx):
        if self._loaded_file_idx != file_idx:
            self._loaded_file = np.load(self.file_names[file_idx])
            self._loaded_file_idx = file_idx
        return self._loaded_file[internal_idx][:, self.mask]

    def __getitem__(self, idx_or_slice):
        if isinstance(idx_or_slice, int):
            assert abs(idx_or_slice) < len(self), f"Index {idx_or_slice} out of range"
            file_idx, internal_idx = self._idx_in_file(idx_or_slice)
            return self._get_from_file(file_idx, internal_idx)
        if isinstance(idx_or_slice, slice):
            all_required = np.arange(*idx_or_slice.indices(len(self)))
            return self.__getitem__(all_required)
        if isinstance(idx_or_slice, (list, np.ndarray)):
            to_return = np.zeros((len(idx_or_slice), self._n_features))
            found = np.zeros(len(idx_or_slice), dtype=bool)
            start_file, start_internal = self._idx_in_file(idx_or_slice[0])
            stop_file, stop_internal = self._idx_in_file(idx_or_slice[-1])
            return_idx_reached = 0
            for file_idx in range(start_file, stop_file + 1):
                wanted_here = np.where(
                    (idx_or_slice >= self.first_idxs[file_idx])
                    & (idx_or_slice < self.first_idxs[file_idx + 1])
                )[0]
                n_here = len(wanted_here)
                if n_here == 0:
                    continue
                internal_idx = idx_or_slice[wanted_here] - self.first_idxs[file_idx]
                found[return_idx_reached : return_idx_reached + n_here] = True
                to_return[
                    return_idx_reached : return_idx_reached + n_here
                ] = self._get_from_file(file_idx, internal_idx)
                return_idx_reached += n_here
            assert np.all(found), "Some indices not found"
            return to_return
        raise ValueError(
            f"Invalid index type {type(idx_or_slice)}, must be int, slice, list or np.ndarray"
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Dataset:
    def __init__(
        self,
        g4_data_folder,
        generator_data_folder,
        fractional_range,
        events_per_input,
        feature_mask=None,
    ):
        self.feature_mask = feature_mask
        self.events_per_input = events_per_input
        self.g4_features = self.get_features(g4_data_folder)
        self.generator_features = self.get_features(generator_data_folder)
        self._g4_fraction_start = int(
            fractional_range[0] * len(self.g4_features) / events_per_input
        )
        self._g4_fraction_length = int(
            (fractional_range[1] - fractional_range[0])
            * len(self.g4_features)
            / events_per_input
        )
        self._generator_fraction_start = int(
            fractional_range[0] * len(self.generator_features) / events_per_input
        )
        self._generator_fraction_length = int(
            (fractional_range[1] - fractional_range[0])
            * len(self.generator_features)
            / events_per_input
        )
        self.__len = self._g4_fraction_length + self._generator_fraction_length

    def get_features(self, data_folder):
        features = Features(self.feature_mask)
        for file_name in os.listdir(data_folder):
            if not file_name.endswith(".npy"):
                continue
            file_path = os.path.join(data_folder, file_name)
            features.add_file(file_path)
        return features

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        if idx < self._g4_fraction_length:
            local_idx = (idx + self._g4_fraction_start) * self.events_per_input
            label = 0
            features = self.g4_features[local_idx : local_idx + self.events_per_input]
        elif idx < self.__len:
            local_idx = (
                idx - self._g4_fraction_length + self._generator_fraction_start
            ) * self.events_per_input
            label = 1
            features = self.generator_features[
                local_idx : local_idx + self.events_per_input
            ]
        else:
            raise IndexError(f"Index out of range, max is {self.__len}")
        return label, torch.tensor(features.flatten()).float()


# context to force the latest model to save before the context is exited
class CallBeforeExit:
    def __init__(self, function):
        self.function = function

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.function()


class Training:
    evaluation_fraction = 0.2
    validation_fraction = 0.3

    def __init__(
        self,
        label,
        g4_data_folder,
        generator_data_folder,
        descriminator_params,
        feature_mask,
        chatty=True,
    ):
        self.label = label
        self.chatty = chatty
        self.state_dict = {}
        self.state_dict["descriminator_params"] = descriminator_params
        self.state_dict["feature_mask"] = feature_mask
        self.state_dict["g4_data_folder"] = g4_data_folder
        self.state_dict["generator_data_folder"] = generator_data_folder
        # Test dataset
        self.state_dict["test_range"] = (0, self.evaluation_fraction)
        self._test_dataset = Dataset(
            g4_data_folder,
            generator_data_folder,
            self.state_dict["test_range"],
            descriminator_params["events_per_input"],
            feature_mask,
        )
        # Validation dataset
        self.state_dict["validation_range"] = (
            self.evaluation_fraction,
            self.evaluation_fraction + self.validation_fraction,
        )
        self._validation_dataset = Dataset(
            g4_data_folder,
            generator_data_folder,
            self.state_dict["validation_range"],
            descriminator_params["events_per_input"],
            feature_mask,
        )
        self._validation_dataloader = torch.utils.data.DataLoader(
            self._validation_dataset, batch_size=32, shuffle=False
        )
        # Training dataset
        self.state_dict["train_range"] = self.state_dict["validation_range"][1], 1
        self._train_dataset = Dataset(
            g4_data_folder,
            generator_data_folder,
            self.state_dict["train_range"],
            descriminator_params["events_per_input"],
            feature_mask,
        )
        self._train_dataloader = torch.utils.data.DataLoader(
            self._train_dataset, batch_size=32, shuffle=True
        )
        self.state_dict["epochs"] = []
        self.state_dict["validation_scores"] = []
        self.state_dict["train_scores"] = []
        self._latest_model = Discriminator(**descriminator_params)
        self.optimizer = torch.optim.Adam(
            self._latest_model.parameters().values(), lr=1e-4, betas=(0.9, 0.999)
        )
        self.criterion = nn.BCELoss()

    @property
    def stats_save_path(self):
        return os.path.join(
            self.state_dict["generator_data_folder"], f"{self.label}_stats.npy"
        )

    def delete(self):
        for file in [
            self.best_model_path,
            self.latest_model_path,
            self.stats_save_path,
        ]:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        self.state_dict["epochs"] = []
        self.state_dict["validation_scores"] = []
        self.state_dict["train_scores"] = []
        self._latest_model = Discriminator(**self.state_dict["descriminator_params"])

    def save(self):
        save_path = os.path.join(
            self.state_dict["generator_data_folder"], f"{self.label}.pt"
        )
        torch.save(self.state_dict, save_path)
        return save_path

    def _save_latest(self):
        self._latest_model.save(self.latest_model_path)
        self.save()

    def reload(self):
        save_path = os.path.join(
            self.state_dict["generator_data_folder"], f"{self.label}.pt"
        )
        loaded = torch.load(save_path, weights_only=False)
        assert loaded["g4_data_folder"] == self.state_dict["g4_data_folder"]
        assert (
            loaded["generator_data_folder"] == self.state_dict["generator_data_folder"]
        )
        if loaded["feature_mask"] is None:
            assert self.state_dict["feature_mask"] is None
        else:
            assert np.all(loaded["feature_mask"] == self.state_dict["feature_mask"])
        for key in self.state_dict["descriminator_params"]:
            assert (
                loaded["descriminator_params"][key]
                == self.state_dict["descriminator_params"][key]
            )
        self.state_dict = loaded
        self._latest_model.load(self.latest_model_path)

    @classmethod
    def load(cls, file_path):
        label = ".".join(os.path.basename(file_path).split(".")[:-1])
        state_dict = torch.load(file_path, weights_only=False)
        training = cls(
            label,
            state_dict["g4_data_folder"],
            state_dict["generator_data_folder"],
            state_dict["descriminator_params"],
            state_dict["feature_mask"],
        )
        training.state_dict = torch.load(file_path, weights_only=False)
        return training

    @property
    def best_model_path(self):
        return os.path.join(
            self.state_dict["generator_data_folder"], f"best_{self.label}.pt"
        )

    @property
    def latest_model_path(self):
        return os.path.join(
            self.state_dict["generator_data_folder"], f"latest_{self.label}.pt"
        )

    def predict_test(self, model=None):
        if model is None:
            model = Discriminator.load(self.best_model_path)
        targets = np.zeros(len(self._test_dataset))
        outputs = np.zeros(len(self._test_dataset))
        for i, (target, data) in enumerate(self._test_dataset):
            output = model(data)
            targets[i] = target
            outputs[i] = output
        return targets, outputs

    def get_validation_loss(self, model):
        loss = 0
        for label, data in self._validation_dataloader:
            output = model(data)
            loss += self.criterion(output.flatten(), label.float()).item()
        return loss / len(self._validation_dataset)

    def epoch(self):
        train_loss = 0
        for i, (label, data) in enumerate(self._train_dataloader):
            self.optimizer.zero_grad()
            output = self._latest_model(data)
            loss = self.criterion(output.flatten(), label.float())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if self.chatty and i % 100 == 0:
                mean_loss = train_loss / (i + 1)
                print(
                    f"\t Batch {i}/{len(self._train_dataloader)}, "
                    f"mean loss {mean_loss}",
                    end="\r",
                )
        return train_loss / len(self._train_dataset)

    def train(self, n_epochs):
        best_loss = np.min(self.state_dict["validation_scores"], initial=np.inf)
        print(f"Training for {n_epochs} epochs")
        print()
        with CallBeforeExit(self._save_latest):
            for _ in range(n_epochs):
                if self.chatty:
                    print(f"Epoch {len(self.state_dict['epochs'])}\n")
                epoch_num = len(self.state_dict["epochs"])
                print(epoch_num, end="\r")
                train_loss = self.epoch()
                self.state_dict["epochs"].append(epoch_num)
                self.state_dict["train_scores"].append(train_loss)
                validation_loss = self.get_validation_loss(self._latest_model)
                self.state_dict["validation_scores"].append(validation_loss)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    self._latest_model.save(self.best_model_path)
                    self._save_latest()
                elif epoch_num % 200 == 0:
                    self._latest_model.save(self.latest_model_path)
        print()

    def plots(self, axarr=None, colour="black", bins=100, legend_name=""):
        if axarr is None:
            fig, axarr = plt.subplots(1, 2, figsize=(15, 5))
        # first the training progress
        axarr[0].plot(
            self.state_dict["epochs"],
            self.state_dict["train_scores"],
            c=colour,
            ls="--",
            label=f"Train {self.label} {legend_name}",
        )
        axarr[0].plot(
            self.state_dict["epochs"],
            self.state_dict["validation_scores"],
            c=colour,
            ls="-",
            label=f"Validation {self.label} {legend_name}",
        )
        axarr[0].set_xlabel("Epoch")
        axarr[0].set_ylabel("Loss")
        # then the results on the test data
        targets, outputs = self.predict_test()
        outputs_for_0 = outputs[targets == 0]
        outputs_for_1 = outputs[targets == 1]
        if isinstance(bins, int):
            bins = np.linspace(0, 1, bins)
        axarr[1].hist(
            outputs_for_0,
            bins=bins,
            histtype="step",
            facecolor=(0.5, 0.5, 0.5, 0.1),
            hatch="\\",
            label=f"0 {self.label}",
            fill=True,
            edgecolor=colour,
        )
        axarr[1].hist(
            outputs_for_1,
            bins=bins,
            histtype="step",
            facecolor=(0.5, 0.5, 0.5, 0.1),
            hatch="/",
            label=f"1 {self.label}",
            fill=True,
            edgecolor=colour,
        )


def locate_g4_data(configs):
    dataset_name = naming.dataset_name_from_path(configs.dataset_path)
    data_folder = os.path.join(configs.logdir, dataset_name, "discriminator/g4")
    return data_folder


def locate_model_data(configs, model_path):
    dataset_name = naming.dataset_name_from_path(configs.dataset_path)
    model_base_name = ".".join(os.path.basename(model_path).split(".")[:-1])
    data_folder = os.path.join(configs.logdir, dataset_name, "discriminator", model_base_name)
    return data_folder


def get_file_paths(data_folder, n_events):
    os.makedirs(data_folder, exist_ok=True)
    n_events = np.atleast_1d(n_events)
    file_names = [
        os.path.join(data_folder, f"{i}_{n}.npy") for i, n in enumerate(n_events)
    ]
    return file_names


def create_g4_data_files(configs, redo=False):
    # find output location and check if it exists
    n_events = read_write.get_n_events(configs.dataset_path, configs.n_dataset_files)
    n_events = np.atleast_1d(n_events)
    showerflow_dir = showerflow_utils.get_showerflow_dir(configs)
    data_folder = locate_g4_data(configs)
    file_paths = get_file_paths(data_folder, n_events)
    exists = [os.path.exists(file_path) for file_path in file_paths]
    if (not redo) and all(exists):
        print("All g4 data files already exist")
        return
    print(f"Some g4 data files are missing; {sum(exists)}/{len(exists)}")

    # calculate features, of find precalculated features
    clusters_per_layer_path = showerflow_training.get_clusters_per_layer(
        configs, showerflow_dir
    )
    energy_per_layer_path = showerflow_training.get_energy_per_layer(
        configs, showerflow_dir
    )
    cog_path, _ = showerflow_training.get_cog(configs, showerflow_dir)

    # decide on output shape
    data = np.load(clusters_per_layer_path)
    n_layers = data["clusters_per_layer"].shape[1]
    del data
    n_features = 3 + 2 * n_layers

    print("Creating g4 data files")
    print()
    for i, end in enumerate(np.cumsum(n_events)):
        print(f"{i/len(n_events):.0%}", end="\r")
        length = n_events[i]
        start = end - length
        features = np.empty((length, n_features))
        data = np.load(cog_path)
        features[:, :3] = data[start:end]
        del data
        data = np.load(clusters_per_layer_path)
        clusters_per_layer = data["clusters_per_layer"]
        features[:, 3 : 3 + n_layers] = clusters_per_layer[start:end, :]
        del data
        data = np.load(energy_per_layer_path)
        gev_to_mev = 1000
        energy_per_layer = data["energy_per_layer"] * gev_to_mev
        features[:, 3 + n_layers :] = energy_per_layer[start:end, :]
        del data
        np.save(file_paths[i], features)
    print()


def create_showerflow_data_files(configs, model_path, redo=False):
    n_events = read_write.get_n_events(configs.dataset_path, configs.n_dataset_files)
    n_events = np.atleast_1d(n_events)
    data_folder = locate_model_data(configs, model_path)
    file_paths = get_file_paths(data_folder, n_events)
    _, distribution, _ = generate.load_flow_model(configs, model_path)

    # check if this is already done
    exists = [os.path.exists(file_path) for file_path in file_paths]
    if (not redo) and all(exists):
        print(f"All {model_path} data files already exist")
        return
    print(f"Some {model_path} data files are missing; {sum(exists)}/{len(exists)}")

    # sample from shower flow
    print(f"Creating {model_path} data files")
    print()
    local_batch_size = 1000
    total_events = np.sum(n_events)
    for i, end in enumerate(np.cumsum(n_events)):
        length = n_events[i]
        start = end - length
        features = []
        for j in range(start, end, local_batch_size):
            print(f"{j/total_events:.0%}", end="\r")
            batch_end = min(j + local_batch_size, end)
            cond, _ = read_raw_regaxes_withcond(
                configs,
                slice(j, batch_end),
                for_model="showerflow",
            )
            del _
            if len(cond.shape) == 1:
                cond = cond[:, None]
            cond = normalise_cond_feats(configs, cond, for_model="showerflow")
            batch = conditioned_sample(configs, distribution, cond)
            features.append(batch)
        features = np.concatenate(features, axis=0)
        np.save(file_paths[i], features)
        del features
        del cond
    print()


def conditioned_sample(configs, distribution, cond_batch):
    bs = cond_batch.shape[0]
    cond_batch = torch.tensor(cond_batch).float().to(configs.device)
    samples = (
        distribution.condition(cond_batch)
        .sample(
            torch.Size(
                [
                    bs,
                ]
            )
        )
        .cpu()
        .numpy()
    )

    (
        num_clusters,
        energies,
        cog_x,
        cog_y,
        cog_z,
        clusters_per_layer_gen,
        e_per_layer_gen,
    ) = showerflow_utils.truescale_showerflow_output(samples, configs)

    if not getattr(configs, "shower_flow_fixed_input_norms", False):
        # scale relative clusters per layer to actual number of clusters per layer
        # and same for energy
        clusters_per_layer_gen = (
            clusters_per_layer_gen
            / clusters_per_layer_gen.sum(axis=1, keepdims=True)
            * num_clusters
        ).astype(
            int
        )  # B,30
        e_per_layer_gen = (
            e_per_layer_gen / e_per_layer_gen.sum(axis=1, keepdims=True) * energies
        )  # B,30

    n_layers = clusters_per_layer_gen.shape[1]

    output = np.empty((bs, 3 + 2 * n_layers))
    output[:, 0] = cog_x
    output[:, 1] = cog_y
    output[:, 2] = cog_z
    output[:, 3 : 3 + n_layers] = clusters_per_layer_gen
    output[:, -n_layers:] = e_per_layer_gen
    return output
