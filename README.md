UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI
===

# Introduction
**UnrealZoo is a suite of large-scale, photorealistic virtual environments designed to enhance the development of general visual intelligence, particularly for embodied AI. Leveraging the Unreal Engine and UnrealCV, these environments provide a scalable, controllable, and diverse play ground for AI agents, helping to speed up advancements in embodied artificial intelligence. The use of high-fidelity graphics and physics simulations allows for the study of complex interactive visual tasks in open worlds. We outline the technical framework of UnrealCV+, emphasizing its integration with machine learning pipelines and its utility in generating synthetic data for AI research. We demonstrate the applications of the virtual environments in developing the visual intelligence for embodied agents. This project concludes with a perspective on the future integration of multi-modal inputs and the role of UnrealCV+ in shaping the next generation of AI systems with advanced visual intelligence.**

To accommodate users' local memory limitations, we provide two different lightweight environment options based on **UE4 and UE5**. The full environment package, which includes 100+ scene maps (>60GB), is also available for download **here**.

**UE4 Example Scenes**
<table>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/track_train.png" width="320" height="180">
        <figcaption> track train</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/Greek_island.png" width="320" height="180">
        <figcaption>Greek Island</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/Containeryard_Day.png" width="320" height="180">
        <figcaption>ContaineYard_Day</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/ContainerYard_Night.png" width="320" height="180">
        <figcaption>ContainerYard_Night</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/SuburbNeighborhood_Day.png" width="320" height="180">
        <figcaption>SuburbNeighborhood_Day</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/SuburbNeighborhood_Night.png" width="320" height="180">
        <figcaption>SuburbNeighborhood_Night</figcaption>
      </figure>
    </td>
  </tr>
</table>

**UE5 Example Scenes**
<table>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/ChemicalFactory.png" width="320" height="180">
        <figcaption> ChemicalFactory</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/ModularOldTown.png" width="320" height="180">
        <figcaption>Modular Old Town</figcaption>
      </figure>
    </td>

  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/MiddleEast.png" width="320" height="180">
        <figcaption>MiddleEast</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE5_ExampleScene/Roof-City.png" width="320" height="180">
        <figcaption>Roof-City</figcaption>
      </figure>
    </td>
 
  </tr>
</table>

# FrameWork
![framework](doc/figs/framework.png)

[//]: # (- ```UnrealCV``` is the basic bridge between ```Unreal Engine``` and ```OpenAI Gym```.)

[//]: # (- ```OpenAI Gym``` is a toolkit for developing an RL algorithm, compatible with most numerical computation libraries, such as TensorFlow or PyTorch. )
- The ```Unreal Engine Environments (Binary)``` contains the scenes and playable entities.
- The ```UnrealCV+ Server``` is built in the UE binary as a plugin, icluding modules for rendering , data capture, object/agent control, and command parsing. We have optimized the rendering pipeline and command system in the server.
- The ```UnrealCV+ Client``` provides Python-based utility functions for launching the binary, connecting with the server, and interacting with UE environments. It uses IPC sockets and batch commands for optimized performance.
- The ```OpenAI Gym Interface``` provides agent-level interface for agent-environment interactions, which has been widely used in the community. Our gym interface supports customizing the task in a configuration file and contains a toolkit with a set of gym wrappers for environment augmentation, population control, etc.


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
## Install Gym-UnrealZoo

It is easy to install gym-unrealzoo, just run
```
git clone https://github.com/UnrealZoo/gym-unrealzoo.git
cd gym-unrealzoo
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
You can load them from the following link:

| Environment                   | Download Link                                                                                                 | Size   |
|-------------------------------|---------------------------------------------------------------------------------------------------------------|--------|
| UE4 Example Scene (Linux)     | [Download](https://app.filen.io/#/d/67ced4e0-4828-4614-9a2d-a9f2360232a5#K7eiIfKYO8GBl7yneiK8kACcRucs5P5O)    | ~3GB   |
| UE4 Example Scene (Windows)   | [Download](https://app.filen.io/#/d/f4bd1676-b739-4703-885e-df96f032bd84#BPXaTn3HFsGjQI5LRP12IUWvhLqwB3Fd)    | ~3GB   |
| UE4 Example Scene (Mac)       | [Download](https://app.filen.io/#/d/a0e7fde8-f183-4cb8-8985-5f8d50b7f9b4#LjsLjrKeZxIjemUhzWp7VnJTulSxlpTy)    | ~3GB   |
| UE5 Example Scene (Linux)     | [Download](https://app.filen.io/#/d/3982f3f5-c21e-400f-b920-0fd7f255a0c1#bgO3LKFfbmISGcYQHuOvliGetWDQMBEc)    | ~10GB  |
| UE5 Example Scene (Windows)   | [Download](https://app.filen.io/#/d/bb512fa8-ac3a-4b96-ab40-770d262b19e7#2Xj8bDjRxVhdlL5RcmE6aZhC4S1VvMSb)    | ~10GB  |
| UE4_Scene_Collection(Linux)   | [Download](https://app.filen.io/#/d/c3e1c06f-9d63-4c6f-8940-55b0812e922b#ZkTVetGo8EaF6TUNQV4rwtbEdFitstGD)    | \>60GB |
| UE4_Scene_Collection(Windows) | [Download](https://drive.filen.io/d/9fc755ac-19b5-4284-9385-9387e9a5ee86#ZzGNVRVV8fyp99VJfPMehsTk0xH7cXcb)    | \>60GB |

[//]: # (`ENV_NAME` can be `RealisticRoom`, `RandomRoom`, `Arm`, etc. )

[//]: # (After that, it will automatically download a related env binary)

[//]: # (to the [UnrealEnv]&#40;gym_unrealcv/envs/UnrealEnv&#41; directory.)

[//]: # ()
[//]: # (**Please refer the ``binary_list`` in [load_env.py]&#40;load_env.py&#41; for more available example environments.**)


