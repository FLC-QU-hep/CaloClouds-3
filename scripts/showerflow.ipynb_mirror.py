import numpy as np
from matplotlib import pyplot as plt
from pointcloud.utils import plotting
import os

directory = "/home/henry/training/point-cloud-diffusion-data/showerFlow/samples/"
files = [os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith("g4")]
g4_file = os.path.join(directory, "g4.npz")
g4_data = np.load(g4_file)
print({k:g4_data[k].shape for k in g4_data.keys()})

