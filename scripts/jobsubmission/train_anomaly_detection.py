"""
Run the anomaly detection, optionally from an existing model.
"""
import sys

from pointcloud.configs import Configs
from pointcloud.anomalies.train import train

config = Configs()
# optionally, first cli arg is the path to a model to load
model_path = sys.argv[1] if len(sys.argv) > 1 else None
train(config, model_path)
