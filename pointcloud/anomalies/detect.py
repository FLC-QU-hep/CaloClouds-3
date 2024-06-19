"""
Run the detector and save anomaly scores for each event
"""

import torch
import numpy as np
from matplotlib import pyplot as plt

from .train import TreeDataset, get_criterion
from .autoencoder import GraphAutoencoder
from ..utils.plotting import plot_event
from ..data.read_write import read_raw_regaxes


def save_path_for_model(model_path):
    """
    Path to save the anomaly scores
    """
    ending = model_path.split(".")[-1]
    assert ending.startswith("p")
    save_path = model_path.replace("." + ending, "_scores.npy")
    return save_path


def score_data(model_path, dataset, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(dataset)
    model = GraphAutoencoder.load(model_path)
    model.eval()
    criterion = get_criterion()
    scores = np.empty((end_idx - start_idx, 2))
    scores[:, 0] = np.arange(start_idx, end_idx)
    with torch.no_grad():
        for row, idx in enumerate(scores[:, 0]):
            data = dataset[int(idx)]
            reconstruction = model(data)
            loss = criterion(reconstruction, data.x)
            scores[row, 1] = loss.item()
    return scores


def detect(model_path, configs, batch_size=1000):
    """
    Run over the data in batches, calculate the anomaly scores and save them
    """
    save_path = save_path_for_model(model_path)
    dataset = TreeDataset(configs)
    total_datapoints = len(dataset)
    scores = []
    for start_idx in range(0, total_datapoints, batch_size):
        print(f"{start_idx/total_datapoints:.2%}", end="\r", flush=True)
        end_idx = min(start_idx + batch_size, total_datapoints)
        scores.append(score_data(model_path, dataset, start_idx, end_idx))
        np.save(save_path, np.vstack(scores))
    np.save(save_path, np.vstack(scores))
    return save_path


def get_rating():
    rating = input("Rate the event from 0-10. Press q to quit. ")
    if rating == "q":
        return None
    try:
        rating = float(rating)
        if 0 <= rating <= 10:
            return rating
    except ValueError:
        pass
    print("Invalid rating. Please try again.")
    return get_rating()


def draw_batch(configs, idxs, scores, unrated, batch_length=100):
    unrated = np.copy(unrated)
    min_score = scores.min()
    max_score = scores.max()
    pick_idxs = []
    close_to = np.random.random(batch_length) * (max_score - min_score) + min_score
    for location in close_to:
        closest = np.argmin(np.abs(scores[unrated] - location))
        pick_idxs.append(idxs[unrated][closest])
        unrated[np.where(unrated)[0][closest]] = False
        if not np.any(unrated):
            break
    pick_idxs = sorted(pick_idxs)
    incidents, events = read_raw_regaxes(configs, pick_idxs)
    return list(zip(pick_idxs, incidents, events))


def human_evaulation(model_path, configs):
    # get the scores
    score_path = save_path_for_model(model_path)
    idxs, scores = np.load(score_path).T
    idxs = idxs.astype(int)
    n_scored = scores.shape[0]

    # array to score the ratings in
    ratings = np.full(n_scored, np.nan)
    unrated = np.ones(n_scored, dtype=bool)
    
    input("Each time you are shown an event, rate it for anomaly on "
          "a scale of 0-10. Press enter to continue."
          "Press q to quit. Press any button to start.")
    batch = []
    plt.ion()
    while np.any(unrated):
        if not batch:
            batch = draw_batch(configs, idxs, scores, unrated)
        for idx, incident, event in batch:
            plot_event(idx, incident, event, energy_scale=1000)
            plt.show()
            rating = get_rating()
            plt.close()
            if rating is None:
                break
            mask = idxs == idx
            ratings[mask] = rating
            unrated[mask] = False
        if rating is None:
            break
    results = np.vstack((idxs, scores, ratings)).T
    np.save(score_path.replace(".npy", "_ratings.npy"), results)
    return results

