# Error Catching
- RuntimeError: Fail to initialize OpenGL: enter the following command in the terminal
```bash
unset LD_PRELOAD
```

- AttributeError: module 'distutils' has no attribute 'version': `pip install setuptools==59.5.0`

- mujoco rendering is not using gpu: check this [link](https://github.com/openai/mujoco-py/issues/493)

- `AssertionError: Default values can only be a CfgNode or {<class 'int'>, <class 'bool'>, <class 'str'>, <class 'float'>, <class 'list'>, <class 'tuple'>}`: make sure yacs version, and `pip install yacs==0.1.8`

- `AttributeError: Can't get attribute '_make_function' on <module 'cloudpickle.cloudpickle' `: make sure cloudpickle version, and `pip install cloudpickle --upgrade`

- `*** SystemError: initialization of _internal failed without raising an exception`:
```bash
pip uninstall numba
pip install -U numba
```

- `File "/home/yanjieze/miniconda3/envs/dex/lib/python3.9/site-packages/torch/utils/cpp_extension.py", line 1649, in verify_ninja_availability
    raise RuntimeError("Ninja is required to load C++ extensions")
RuntimeError: Ninja is required to load C++ extensions`
```bash
sudo apt-get install ninja-build
```

- error when compiling mujoco-py 
```
Error compiling Cython file:
------------------------------------------------------------
...
    See c_warning_callback, which is the C wrapper to the user defined function
    '''
    global py_warning_callback
    global mju_user_warning
    py_warning_callback = warn
    mju_user_warning = c_warning_callback
                       ^
------------------------------------------------------------
```
solution
```bash
pip install Cython==0.29.35
```

- error when compiling mujoco-py f
```
2.1.2.14/mujoco_py/gl/eglshim.c:4:10: fatal error: GL/glew.h: No such file or directory
    4 | #include <GL/glew.h>
      |          ^~~~~~~~~~~
compilation terminated.
```
solution
```bash
sudo apt-get install libglew-dev
```

- *** RuntimeError: CUDA error: no kernel image is available for execution on the device
this occurs when using the functions from pytorch3d. reinstalling pytorch3d would solve the problem.
```
pip uninstall pytorch3d
cd third_party/pytorch3d
pip install -e .
cd ../..
```


- ImportError: cannot import name '_C' from 'pytorch3d' (/home/yanjieze/projects/diffusion-policy-for-dex/third_party/pytorch3d/pytorch3d/__init__.py)
```
pip uninstall pytorch3d
cd third_party/pytorch3d
pip install -e .
cd ../..
```

- wandb video error:
```
X Error of failed request:  BadAccess (attempt to access private resource denied)
  Major opcode of failed request:  152 (GLX)
  Minor opcode of failed request:  5 (X_GLXMakeCurrent)
  Serial number of failed request:  112
  Current serial number in output stream:  112
```
solution:
```bash
export QT_GRAPHICSSYSTEM=native
```


- ImportError("/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/gu/anaconda3/lib/python3.9/site-packages/pandas/_libs/window/aggregations.cpython-39-x86_64-linux-gnu.so)")
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install gcc-4.9
sudo apt-get upgrade libstdc++6
```
check by:
```
strings /usr/lib/arm-linux-gnueabihf/libstdc++.so.6 | grep GLIBCXX
```

