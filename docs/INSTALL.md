# Installing Conda Environment from Zero to Hero

The following guidance works well for a machine with 3090/A40/A800/A100 GPU, cuda 11.7, driver 515.65.01.

First, git clone this repo and `cd` into it.
```
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git
```

**Please strictly follow the guidance to avoid any potential errors. Especially, make sure Gym version is the same.**

**Don't worry about the gym version now. Just install my version in `third_party/gym-0.21.0` and you will be fine.**

# 0 create python/pytorch env
```
conda remove -n dex --all
conda create -n dex python=3.8
conda activate dex
```

# 1 install some basic packages
```
pip3 install torch==2.0.1 torchvision torchaudio
# or for cuda 12.1
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install --no-cache-dir wandb ipdb gpustat visdom notebook mediapy torch_geometric natsort scikit-video easydict pandas moviepy imageio imageio-ffmpeg termcolor av open3d dm_control dill==0.3.5.1 hydra-core==1.2.0 einops==0.4.1 diffusers==0.11.1 zarr==2.12.0 numba==0.56.4 pygame==2.1.2 shapely==1.8.4 tensorboard==2.10.1 tensorboardx==2.5.1 absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image yapf==0.31.0 opencv-python==4.5.3.56 psutil av matplotlib setuptools==59.5.0

cd third_party
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ../..

```

# 2 install DP3
```bash
cd third_party/robomimic-0.2.0
pip install -e .
cd ../..

cd 3D-Diffusion-Policy
pip install -e .
cd ..

```

# 3 install environments
Install Adroit environments:
```
pip install --no-cache-dir patchelf==0.17.2.0
cd third_party


cd gym-0.21.0
pip install -e .
cd ..

cd rrl-dependencies
pip install -e mj_envs/.
pip install -e mjrl/.
cd ..
```

install mujoco in `~/.mujoco`
```
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

tar -xvzf mujoco210.tar.gz
```
and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl

```
and then install mujoco-py (in the folder of `third_party`):
```
cd YOUR_PATH_TO_THIRD_PARTY
cd mujoco-py-2.1.2.14
pip install -e .
cd ../..
```


Install DexArt environments:
```bash
cd third_party/dexart-release
pip install -e .
cd ../..
```

download assets from [Google Drive](https://drive.google.com/file/d/1JdReXZjMaqMO0HkZQ4YMiU2wTdGCgum1/view?usp=sharing) and put it in `third_party/dexart-release/assets`.

Install MetaWorld environments:
```bash
cd third_party/Metaworld
pip install -e .
cd ../..
```





# 4 Install our visualizer for pointclouds (optional)
```
pip install -U kaleido
pip install plotly
cd visualizer
pip install -e .
cd ..
```
