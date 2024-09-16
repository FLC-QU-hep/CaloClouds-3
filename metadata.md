# Metadata

What is in the `pointcloud/metadata` folder?
It's data about datasets.
The aim is to minimise the amount of "magic numbers" (i.e. hard coded numbers) relating to datasets, because we all use different datasets.

Is is organised as follows.

## 1. Subfolder

All datasets must have a file name, let's say we are working with the (pretend) dataset `/beegfs/desy/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/gun_henry.hdf5`
then the filename is `gun_henry`. 
This determines which subfolder the metadata will be drawn from.
In this case `pointcloud/metadata/gun_henry`

One exception to this is the letter 'W' in a subfolder name which means 'Wildcard'. Sometimes we have a sequence of datasets with very similar names,
for example `gun_seed1_var.hdf5`, `gun_seed2_var.hdf5`, `gun_seed3_var.hdf5` etc, and we know they all have identical metadata.
Their metadata can be in a subfolder called `pointcloud/metadata/gun_seedW_var`. When the code looks for metadata that matches your filename
the 'W' in the subfolder name will expand as a wildcard, e.g. `pointcloud/metadata/gun_seed*_var`.

## 2. Folder contents

That folder contains a set of numpy files (all ending `.npy`, don't use `.npz`).
Each file either contains just one item, or a pickled dict.
These items describe the data. Below I will go through the data description in the `gun_henry` folder, and explain what each part means.
One key point is that by default, everything is described in the *physical* coordinates of the detector, but
it may be rescaled/translated before it's saved to disc, and the coordinates it's saved to disc with are referred to as `hdf5` (reference to the file format).

The term `global` indicates that a value is given in the detector coordinate system, as opposed to local coordinates, where the shower evolves along the z axis.
Local coordinates should be understood to be the default.

When the `hdf5` label is used, then this implies "global" iff `orientation` is "local:xyz==global:xyz"

**Individual files;**

- `gun_xyz_pos_global`: `[   0.  1804.7  -50. ]` Position of the particle gun in global physical coordinates.
- `gun_xyz_pos_hdf5`: `[  0.  0. -50.]` Position of the particle gun in the coordinates the data is saved in. This may be rescaled from physical units.
- `layer_bottom_pos_global`: `[1811.34020996 1814.46508789 1823.81005859 1826.93505859 1836.2800293 ... ]` y-coordinate of the bottom of each layer, physical coordinates.
- `layer_bottom_pos_hdf5`: `[1811.34020996 1814.46508789 1823.81005859 1826.93505859 1836.2800293 ... ]` Same as above, but saved coordinates.
- `orientation_global`: `"hdf5:xyz==global:xyz"` the relationship between the coordinates of the clusters saved to disk, and the global detector coordinates. Ideally, it's always `"hdf5:xyz==global:xyz"`, but if `local_xyz_orientation` is `True` then it is likely something else as the dataset was rotated after generating but before saving.
- `orientation`: `"hdf5:xyz==local:yzx"` the relationship between the local shower frame coordinates, and the coordinates saved in the hdf5 file. Shower moves along the z axes in local coordinates. Determines which direction the shower is moving in the save file, should be possible to verify by looking at the data.


**Then there are some files that contain pickled dictionaries;**

Box edges, only data inside a cuboid around the shower is recorded from the G4 simulation. These values give the limits, in physical coordinates, of that cube.

- `Ymin_global`: `1811` 
- `Xmin_global`: `-200`
- `Xmax_global`: `200`
- `Zmin_global`: `-260`
- `Zmax_global`: `140`

Cell dimensions and hdf5 cell dimensions.

- `half_cell_size_global`: `2.5441665649414062` Half the physical cell size (axis parallel to detector circumference)
- `half_cell_size_hdf5`: `2.5441665649414062` Same as previous, but in the units the data is saved in.
- `cell_thickness_global`: `0.5250244140625` The thickness of the cell in physical dimensions in the radial direction.
- `cell_thickness_hdf5`: `0.5250244140625` Same as previous, but in the units the data is saved

Downsample settings, choices made when the data was created. See [https://gitlab.desy.de/ftx-sft/generative/data_production/-/blob/main/highGran_downsample.py](https://gitlab.desy.de/ftx-sft/generative/data_production/-/blob/main/highGran_downsample.py). Some of them I don't know the meaning of, but it could probably be inferred from the linked code.

- `y_scale_is_layer_n`: `False` ??
- `all_steps`: `False`  Are all steps from G4 preserved?
- `detector_grid`: `False` ??
- `cog_in_cell`: `False` ??
- `dm`: `5` How many subsections each physical calorimeter cell is divided into along each dimension. 
- `grid_pos_fixed`: `True` ??
- `sort`: `False` Should the events be sorted by number of hits before saving.
- `aligne`: `False` Should the centre of energy of each layer be aligned with (0, 0) before saving.
- `local_xyz_orientation`: `False`, if `True` then shower evolves along the z axis. If not, it depends on the  check `hdf5_orientation` to see what the relationship between local and global coordinates.

Rescales, used when learning distributions like showerflow.

- `incident_rescale`: `100` 
- `vis_eng_rescale`: `2.5`
- `mean_cog_hdf5`: `[ 3.80404145e-01  1.89524914e+03 -4.99723479e+01]`
- `std_cog_hdf5`: `[ 0.51969929 12.76039063  0.48370289]`
- `n_pts_rescale`: `5000`

If any of these don't make much sense, it might be a good idea to look at where they are used in the code to get an understanding of them.

## 3. Interface

Naturally, there is a common interface to help read this data in your code.
Below is an example of generating this interface and using it.

```python
from pointcloud.config_varients import caloclouds_3  # chose the apropreate configs
configs = caloclouds_3.Configs()

print(f"We are using the dataset at {configs.dataset_path}")

```

Which prints something like;
> We are using the dataset at `/beegfs/desy/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/gun_henry.hdf5`

The `.dataset_path` attribute of the configs determines which `pointcloud/metadata` subfolder is read from.

```python
from pointcloud.utils.metadata import Metadata  # import the interface

# now create an interface
metadata = Metadata(configs)
# the metadata has an attribute for each of the bullet points in the previous section
print(f"The half_cell_size_hdf5 is {metadata.half_cell_size_hdf5} and " +
      f"the bottom y coordinates of the first 3 layers are {metadata.layer_bottom_pos_hdf5[:3]}")
```

Which prints something like;
> The half_cell_size_hdf5 is 2.5441665649414062 and the bottom y coordinates of the first 3 layers are [1811.34020996 1814.46508789 1823.81005859]

Sometimes `Metadata` objects will be created in functions from a provided config,
so make sure your `configs.dataset_path` is correct.
This allows functions to avoid magic numbers relating to the dataset, and "just work" on new datasets.


## 4. Extending; what to do when you have a new dataset.

If you are adding a new dataset, you will need to create a new folder in `pointcloud/metadata` with the name of the dataset.
Then you will need to add the appropriate `.npy` files, just like the ones in `pointcloud/metadata/gun_henry`.
For example, let's say we have a new dataset at the path `/beegfs/desy/user/dayhallh/data/p2122_wide_angle.hdf5`

To start with make the relevant subfolder;

```bash
import os
os.mkdir("pointcloud/metadata/p2122_wide_angle")
```

Then use numpy to add the parameters of your dataset;

```python
import numpy as np
np.save("pointcloud/metadata/gun_xyz_pos_global.npy", [0., 1804.7, -50.])  # saved as a seperate file
np.save("pointcloud/metadata/gun_xyz_pos_hdf5.npy", [0., 0., -1.])
box_edges = {"Ymin_global": 1811, "Xmin_global": -200, "Xmax_global": 200, "Zmin_global": -260, "Zmax_global": 140}
np.save("pointcloud/metadata/box_edges.npy", box_edges)  # will get saved as a pickled dict

# ... plus any other parameters you have
```

If you don't know the values of all of the parameters from section 2, don't worry.
Just don't save the values you don't know.
The corresponding attributes just won't be created in the `Metadata` object, and
if they are needed, you will see an `AttributeError` when you run the code.
Should you get an `AttributeError`, you might need to find out what the corresponding value actually is...
or write another case for your data or if you're convinced it won't matter, just put a dummy number in there.

Once you have saved all the numpy files, you should check that it worked by creating the `Metadata` interface and
looking at the values;

```python
from pointcloud.config_varients.wide_angle import Configs
configs = Configs()
# check that the configs point to the right dataset, or it won't open the right metadata subfolder.
assert configs.dataset_path == "/beegfs/desy/user/dayhallh/data/p2122_wide_angle.hdf5"

from pointcloud.utils.metadata import Metadata
metadata = Metadata(configs)
# the content of the npy files you just saved are the attributes of the metadata.
# where a file contains a simple array/float, the attribute name is the file name
print(metadata.gun_xyz_pos_hdf5)
# where a file contained a pickled dict (like box_edges.npy) then the keys of the dict
# become attribute names.
print(metadata.Xmin_global)
# You can see everything at once using repr
print(repr(metadata))
```


### New metadata parameters

What if your data has more parameters?
Say you fire the gun multiple times per event; you might call the corrisponding variable `n_times_fired`.
Adding it to your metadata is very simple; just save a new numpy file in the subfolder.
When the `Metadata` object is created, it will read all the files ending in `.npy` in the subfolder, including new ones.
(N.B. it will not read `.npz` files)

```python
np.save("pointcloud/metadata/wide_angle/n_times_fired.npy", 5)

metadata = Metadata(configs)
print(f"Our new metadata is {metadata.n_times_fired}")
```
We expect to see
> Our new metadata is 5

Done ;) feel free to contact me with any questions.
