Gym-UnrealZOO: 
===

# Introduction
**This project integrates Unreal Engine with OpenAI Gym for visual reinforcement learning based on [UnrealCV](http://unrealcv.org/).**
In this project, you can run (Multi-Agent) Reinforcement Learning algorithms in various realistic UE4 environments easily without any knowledge of Unreal Engine and UnrealCV.

A number of environments have been released for robotic vision tasks, including  `Active object tracking`, `Searching for objects`, and `Robot arm control`.

<table>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/track/urbancity.gif" width="240" height="180">
        <figcaption>Tracking in UrbanCity with distractors</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/track/garden.gif" width="240" height="180">
        <figcaption>Tracking in Garden</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/track/snow.gif" width="240" height="180">
        <figcaption>Tracking in SnowForest</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/track/garage.gif" width="240" height="180">
        <figcaption>Tracking in Garage with distractors</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/search/search2.gif" width="240" height="180">
        <figcaption>Searching in RealisticRoom</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/arm/reach-virtual.gif" width="240" height="180">
        <figcaption>Robot Arm Control</figcaption>
      </figure>
    </td>
  </tr>
</table>

The framework of this project is shown below:
![framework](doc/figs/framework.JPG)

- ```UnrealCV``` is the basic bridge between ```Unreal Engine``` and ```OpenAI Gym```.
- ```OpenAI Gym``` is a toolkit for developing an RL algorithm, compatible with most numerical computation libraries, such as TensorFlow or PyTorch. 



# Installation
## Dependencies
- UnrealCV
- Gym
- CV2
- Matplotlib
- Numpy
- Docker(Optional)
- Nvidia-Docker(Optional)
 
We recommend you use [anaconda](https://www.continuum.io/downloads) to install and manage your Python environment.
```CV2``` is used for image processing, like extracting object masks and bounding boxes. ```Matplotlib``` is used for visualization.
## Install Gym-UnrealCV

It is easy to install gym-unrealcv, just run
```
git clone https://github.com/zfw1226/gym-unrealcv.git
cd gym-unrealcv
pip install -e . 
```
While installing gym-unrealcv, dependencies including [OpenAI Gym](https://github.com/openai/gym), unrealcv, numpy and matplotlib are installed.
`Opencv` should be installed additionally. 
If you use ```anaconda```, you can run
```
conda update conda
conda install --channel menpo opencv
```
or
```
pip install opencv-python
```

## Prepare Unreal Binary
Before running the environments, you need to prepare unreal binaries. 
You can load them from clouds by running [load_env.py](load_env.py)
```
python load_env.py -e {ENV_NAME}
```
`ENV_NAME` can be `RealisticRoom`, `RandomRoom`, `Arm`, etc. 
After that, it will automatically download a related env binary
to the [UnrealEnv](gym_unrealcv/envs/UnrealEnv) directory.

**Please refer the ``binary_list`` in [load_env.py](load_env.py) for more available example environments.**


