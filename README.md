![Run python tests](https://github.com/FLC-QU-hep/point-cloud-diffusion/actions/workflows/ci.yml/badge.svg)


# Rolling implementation of CaloClouds and friends

Spawned the CaloClouds model published in ([arXiv:2305.04847](https://arxiv.org/abs/2305.04847)).
and also *CaloClouds II: Ultra-Fast Geometry-Independent Highly-Granular Calorimeter Simulation* ([arXiv:2309.05704](https://arxiv.org/abs/2309.05704)).


Not sure how to do something with git or gitlab? Please start by checking the docs here; [`gitlab.desy.de/ftx-sft/Documentation/-/tree/master/git`](https://gitlab.desy.de/ftx-sft/Documentation/-/tree/master/git)

---

Configs are stored in [`configs/`](./configs/), and the one in use is determined by what the soft link `configs.py` is pointing to.
To make a new config;
- Make the base file with `cp configs/default.py configs/my_funky_new_config_name.py`.
- Edit the funky new config to your taste.
- Remove the existing symbolic link with `rm configs.py`
- Link your config `ln -s configs/my_funky_new_config_name.py configs.py`

The teacher model is trained using [`main.py`](./main.py), and the student model is trained with [`cd.py`](./cd.py).

The Shower Flow (to predict energy and hist per layer) is trained via the notebook [`ShowerFlow.ipynb`](./ShowerFlow.ipynb).

The polynomial fits for the occupancy calculations are performed in [`occupancy_scale.ipynb`](./occupancy_scale.ipynb).

An outline of the sampling process for both CaloClouds II and CaloClouds II (CM) can be found in [`generate.py`](./evaluation/generate.py).

The timing of the models is benchmarked with [`timing.py`](./timing.py)

---

The training dataset is available under the link: https://syncandshare.desy.de/index.php/s/XfDwx33ryERwPdi

But if you are connected to `beegfs` you can also find the relevant goodies in Anatolii's directories.
For example;
`/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5`

---

Code references:
- The code for training the score-based model is based on: https://github.com/crowsonkb/k-diffusion
- The consistency distillation is based on: https://github.com/openai/consistency_models/
- The PointWise Net is adapted from: https://github.com/luost26/diffusion-point-cloud
- Code base for our CaloClouds (1) model: https://github.com/FLC-QU-hep/CaloClouds
