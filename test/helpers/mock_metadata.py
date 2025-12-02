"""
Quickly create and remove symlinks to the apreporeate metadata folder.
This is handled as a context decorator.
"""
import os
from pointcloud.utils.metadata import get_metadata_folder
from pointcloud.config_varients import default
from contextlib import ContextDecorator


class TemporaryMetadata(ContextDecorator):
    def __init__(self, *name_base, target_folder="gun_henry"):
        config = default.Configs()
        config.dataset_path = f"./{target_folder}.h5"
        self.target_folder = get_metadata_folder(config)
        metadata_base = os.path.dirname(self.target_folder)
        self.temp_folders = []
        for base in name_base:
            self.temp_folders.append(os.path.join(metadata_base, base))

    def __enter__(self):
        for temp_folder in self.temp_folders:
            os.symlink(self.target_folder, temp_folder)
        return self

    def __exit__(self, *exc):
        for temp_folder in self.temp_folders:
            os.unlink(temp_folder)
        return False
