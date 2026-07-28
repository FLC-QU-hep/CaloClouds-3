#!/bin/bash

cd /eos/user/m/mamozzan/CaloClouds-3/ || exit
source caloclouds3/bin/activate

# Link your config file to configs.py
cd pointcloud/
rm configs.py
# ln -s config_varients/caloclouds_3_S2P_steps.py configs.py
# ln -s config_varients/caloclouds_3_S2P_withincell.py configs.py
# ln -s config_varients/caloclouds_3_S2P_subcell.py configs.py
ln -s config_varients/caloclouds_3_S2P_subcell_6kcut.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms8_mcs40.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms3_mcs10.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms12_mcs12.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms40_mcs40.py configs.py
cd ..

python scripts/training/ShowerFlow.py

# polynomial fit 
python scripts/training/calcluate_coef.ipynb 

exit