 #echo mamba create -p $dust/envs/calogpu python=3.12
 #mamba create -p $dust/envs/calogpu python=3.12
 #echo mamba_calogpu 
 #mamba_calogpu 
 echo TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ setuptools==69.0.3
 TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ setuptools==69.0.3
 echo TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 echo TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ -r ~/training/point-cloud-diffusion/requirements.txt
 TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ -r ~/training/point-cloud-diffusion/requirements.txt
 #echo TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ pytest
 #TMPDIR=$dust/tmp pip install -U --cache-dir /data/dust/user/dayhallh/pip_cache/ pytest
