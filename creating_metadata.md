If things are the same as `10-90GeV_highGran_fixedAng_05.2024` just write "same" and I'll copy from there.

Some values are needed in physical coordinates, then again in "raw" coordinates, i.e. the coordinate system used to save them on the disk.

- `gun_xyz_pos`: Vector length 3. Position of the particle gun in physical coordinates.
- `gun_xz_pos_raw`: Vector length 2. Position of the particle gun in the coordinates the data is saved in. This may be rescaled from physical units.
- `layer_bottom_pos`: Vector, length n_layers. y-coordinate of the bottom of each layer, physical coordinates.
- `layer_bottom_pos_raw`: Vector length n_layers. Same as above, but saved coordinates.

Box edges, only data inside a cuboid around the shower is recorded from the G4 simulation. These values give the limits, in physical coordinates, of that cube.
- `Ymin`: float
- `Xmin`: float
- `Xmax`: float
- `Zmin`: float
- `Zmax`: float

Cell dimensions and raw cell dimensions.

- `half_cell_size`: float. Half the physical cell size (axis parallel to detector circumference)
- `half_cell_size_raw`: float. Same as previous, but in the units the data is saved in.
- `cell_thickness`: float. The thickness of the cell in physical dimensions in the radial direction.
- `cell_thickness_raw`:  float. Same as previous, but in the units the data is saved

Downsample settings, choices made when the data was created. See [https://gitlab.desy.de/ftx-sft/generative/data_production/-/blob/main/highGran_downsample.py](https://gitlab.desy.de/ftx-sft/generative/data_production/-/blob/main/highGran_downsample.py). 

- `all_steps`: bool. Are all steps from G4 preserved?
- `dm`:  int. How many subsections each physical calorimeter cell is divided into along each dimension. 
- `sort`:  bool. Have the events been sorted by number of hits before saving.
- `aligne`:  bool. Is the center of energy of each layer be aligned with (0, 0) before saving.
- `local_xyz_orientation`: bool. Something about local coordinates? It looks like eventually we might need this to identify which is the y-coord in the data.

Rescales, used when learning distributions like showerflow, or doing inference with them.

- `incident_rescale`: float. What to divide the incident energy by before giving to showerflow.
- `vis_eng_rescale`: float. What to divide the visible energy by before giving to showrflow.
- `mean_cog`: vector length 3. Mean value for the center of gravity in [x,y,z]
- `std_cog`: vector length 3. Std of the center of gravity in [x, y, z]. Used with `mean_cog` to rescale the distribution to unit scale.
- `n_pts_rescale`: float. What to divide the number of clusters by before giving to showerflow.

