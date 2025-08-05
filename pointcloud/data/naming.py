import os


def dataset_name_from_path(dataset_path):
    dataset_name_key = ".".join(os.path.basename(dataset_path).split(".")[:-1])
    if "{" in dataset_name_key:
        dataset_name_key = dataset_name_key.split("{")[0]
    if "seed" in dataset_name_key:
        dataset_name_key = dataset_name_key.split("seed")[0]
    if "file" in dataset_name_key:
        dataset_name_key = dataset_name_key.split("file")[0]
    dataset_name_key = dataset_name_key.strip("_")
    return dataset_name_key
