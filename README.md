UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI
===
![framework](doc/figs/overview.png)

# 🔥 News
- Paper Link: [UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI]()
- Comprehensive documentation is available in [Doc]() 
- Project website is available at [UnrealZoo](https://unrealzoo.github.io/)



# Table of Contents

1. [What is UnrealZoo?](#what-is-unrealzoo)
   - [Key Features](#key-features)
   - [Framework](#framework)
2. [Quick Installation](#quick-installation)
   - [Dependencies](#dependencies)
   - [Install UnrealZoo-Gym](#install-gym-unrealzoo)
   - [Prepare UE Binary](#prepare-ue-binary)
3. [Run the Example Code](#run-the-example-code)
4. [Build your agent](#build-your-agent)
5. [Documentation](#documentation)
6. [TODO List](#todo-list)
7. [License and Acknowledgments](#license-and-acknowledgments)
8. [Citation](#citation)


## What is UnrealZoo?

UnrealZoo is a rich collection of photo-realistic 3D virtual worlds built on Unreal Engine, designed to reflect the complexity and variability of the open worlds. 
There are various playable entities for embodied AI, including human characters, robots, vehicles, and animals.
Integrated with [UnrealCV](https://unrealcv.org/), UnrealZoo provides a suite of easy-to-use Python APIs and tools for various potential applications, such as data annotation and collection, environment augmentation, distributed training, and benchmarking agents.

**💡This repository provides the gym interface based on UnrealCV APIs for UE-based environments, which is compatible with OpenAI Gym and supports the high-level agent-environment interactions in UnrealZoo.**

### Key Features
- **Photorealistic**: High-quality graphics rendering empowered by Unreal Engine (4.27/5.4).
- **Large-scale**: 100+ Scenes, the largest one covers 16km².
- **Diverse Scenes**: landscape🏜️🏞️🏝️, historic architecture⛩️🛕🏰, settlement🏘️🏙️, industrial zone🏭🏗️, facilities🤽🏻‍♂️🚉🏪...
- **Diverse Bodies**: human🚶🏻‍♂️️🚶🏻, robot🤖, vehicles🚗🏍️🛩️,animals🐘🐕🐎🐧🐢🐖🐏🐂...
- **Diverse Actions**: running🏃🏻‍♂️, climbing🧗🏻, sitting🧘🏻, jumping, squatting, pick-up...
- **Easy-to-use**: Pre-build UE binaries, integrated with optimized UnrealCV, are to be used without any prior knowledge of UE.
- **Cross-platform**: Runs on Linux, Windows, macOS
- **Flexible Python APIs**: Provide UnrealCV Python APIs and Gym Interfaces for various potential applications.
- **Multi-agent Interactions**: Real-time interaction of 10+ vision-based agents👩‍👩‍👧‍👧 in one scene.

### Framework
![framework](doc/figs/framework.png)

[//]: # (- ```UnrealCV``` is the basic bridge between ```Unreal Engine``` and ```OpenAI Gym```.)

[//]: # (- ```OpenAI Gym``` is a toolkit for developing an RL algorithm, compatible with most numerical computation libraries, such as TensorFlow or PyTorch. )
- The ```Unreal Engine Environments (Binary)``` contains the scenes and playable entities.
- The ```UnrealCV+ Server``` is built in the UE binary as a plugin, including modules for rendering , data capture, object/agent control, and command parsing. We have optimized the rendering pipeline and command system in the server.
- The ```UnrealCV+ Client``` provides Python-based utility functions for launching the binary, connecting with the server, and interacting with UE environments. It uses IPC sockets and batch commands for optimized performance.
- The ```OpenAI Gym Interface``` provides agent-level interface for agent-environment interactions, which has been widely used in the community. Our gym interface supports customizing the task in a configuration file and contains a toolkit with a set of gym wrappers for environment augmentation, population control, etc.

## Quick Installation
### Dependencies
- UnrealCV
- Gym
- CV2
- Matplotlib
- Numpy
- Docker(Optional)
- Nvidia-Docker(Optional)
 
We recommend you use [anaconda](https://www.continuum.io/downloads) to install and manage your Python environment.
```CV2``` is used for image processing, like extracting object masks and bounding boxes. ```Matplotlib``` is used for visualization.
### Install UnrealZoo-Gym

It is easy to install unrealzoo-gym, just run
```
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
pip install -e . 
```
While installing gym-unrealcv, dependencies including OpenAI Gym, UnrealCV, numpy and matplotlib are installed.

### Prepare UE Binary
Before running the environments, you need to prepare unreal binaries. To accommodate users' local memory limitations, we provide two different lightweight environment options based on **UE4 and UE5**. A larger preview-version environment package, which currently contain 50 scene maps (>60GB), is also available for download **here**. The full version of the UnrealZoo environment package will be released soon.

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
  </tr>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/Containeryard_Day.png" width="320" height="180">
        <figcaption>ContainerYard_Day</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/UE4_ExampleScene/ContainerYard_Night.png" width="320" height="180">
        <figcaption>ContainerYard_Night</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
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

You can load them from the following link:

| Environment                                    | Download Link                                                                                                 | Size       |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------|------------|
| UE4 Example Scene (Linux)                      | [Download](https://app.filen.io/#/d/4f93709c-a0e8-472d-893b-998f80d09a72#mxp3jO4rN0lIlGggK2qMr6m2tj1lMhkv)    | ~3GB       |
| UE4 Example Scene (Windows)                    | [Download](https://app.filen.io/#/d/56adecaa-6888-4514-a8bf-54cd2045d9e3#dHekxfLEPzGymO4Ebu8eEbeXPrCXxLSy)    | ~3GB       |
| UE4 Example Scene (Mac)                        | [Download]()    | ~3GB       |
| UE5 Example Scene (Linux)                      | [Download](https://app.filen.io/#/d/7583151b-68c1-4617-9622-d6ab058ab0ab#mHcE9joY1is8W8lxcNAie6jd6BjR3VYP)    | ~10GB      |
| UE5 Example Scene (Windows)                    | [Download](https://app.filen.io/#/d/00fb2bb7-8b0f-4b5f-8650-087a106fd8f9#O2l0dpqTnadH9CUGhXm8XcbeVDGMaNFt)    | ~10GB      |
| UE4_Scene_Collection_preview(Linux, 50 maps)   | [Download](https://app.filen.io/#/d/c3e1c06f-9d63-4c6f-8940-55b0812e922b#ZkTVetGo8EaF6TUNQV4rwtbEdFitstGD)    | **\>60GB** |
| UE4_Scene_Collection_preview(Windows, 50 maps) | [Download](https://drive.filen.io/d/9fc755ac-19b5-4284-9385-9387e9a5ee86#ZzGNVRVV8fyp99VJfPMehsTk0xH7cXcb)    | **\>60GB** |

Then unzip and move the downloaded binary to the `UnrealEnv` folder, which is our default location for binaries, the folder structures are as follows:
```
gym-unrealcv/  
|-- docs/                  
|-- example/                
|-- gym_unrealcv/              
|   |-- envs/    
|   |   |-- agent/     
|   |   |-- UnrealEnv/                    # Binary default location
|   |   |   |-- Collection_WinNoEditor/   # Binary folder
|   |-- setting/
|   |   |-- env_config/                   # environment config json file location  
...
generate_env_config.py                    # generate environment config json file
...

```

#### Available Map Name in Exemplar Binary

| UE4                     | UE5                     |
|-------------------------|-------------------------|
| track\_train            | Map\_ChemicalPlant\_1   |
| Greek\_Island           | Old\_Town               |
| ContainerYard\_Day      | MiddleEast              |
| ContainerYard\_Night    | Demo\_Roof              |
| SuburbNeighborhood\_Day |                         |
| SuburbNeighborhood\_Night|                        |

#### Naming rule for the environment
We have predefined a naming rule to launch different environment maps and their corresponding task interfaces.  
```Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version} ```
- ```{task}```: the name of the task, we currently support: ```Track```,```Navigation```,```Rendezvous```.
- ```{MapName}```: the name of the map you want to run, ```track_train```, ```Greek_Island```, etc.
- ```{ActionSpace}```: the action space of the agent, ```Discrete```, ```Continuous```, ```Mixed```. (Only Mixed type support interactive actions)
- ```{ObservationType}```: the observation type of the agent, ```Color```, ```Depth```, ```Rgbd```, ```Gray```, ```CG```, ```Mask```, ```Pose```,```MaskDepth```,```ColorMask```.
- ```{version}```: works on ```track_train``` map, ```0-5``` various the augmentation factor(light, obstacles, layout, textures).

## Run the Example Code
#### Hint 💡 
- If your mouse cursor disappears after the game launches, press ``` ` ``` (the key above Tab) to release the mouse cursor from the game.

#### 1. Run random agents
User could choose a map from the available map list, and run the random agent to interact with the environment.
```
python ./example/random_agent_multi.py -e UnrealTrack-track_train-ContinuousColor-v0
```
#### 2. Run a rule-based tracking agent 
```
python ./example/tracking_demo.py -e UnrealTrack-Greek_Island-ContinuousColor-v0
```

#### 3. Run a keyboard tracking agent 
Use the "I", "J", "K", and "L" keys to control the agent's movement.
```
python ./example/keyboard_agent.py -e UnrealTrack-Old_Town-MixedColor-v0
```
#### 4. Run a keyboard Navigation agent
Use the "I", "J", "K", and "L" keys to control the agent's movement.  
Space( ␣ ): Jump.  
Up Arrow(⬆️): Look up.  
Down Arrow(⬇️): Look down.  
(Double "Jump" will trigger the agent to climb)
```
python ./example/keyboard_Navigationagent.py -e UnrealNavigation-Demo_Roof-MixedColor-v0
```
Control the agent to navigate to the target location by using the keyboard. 

<table>
  <tr>
    <td>
      <figure>
        <img src="./doc/figs/navigation/map.png" width="320" height="180">
        <figcaption> Map reference</figcaption>
      </figure>
    </td>
    <td>
      <figure>
        <img src="./doc/figs/navigation/target.png" width="240" height="190
">
        <figcaption>Target Reference</figcaption>
      </figure>
    </td>

  </tr>
 
</table> 

## Build your agent
You can build your agent based on the latest method in reinforcement learning, evaluating it in UnrealZoo.
- Reinforcement Learning
  - Online RL
    - [Visual Navigation Agent]() comming soon..

  - Offline RL
    - [Embodied Visual Tracking Agent(ECCV24)](https://github.com/wukui-muc/Offline_RL_Active_Tracking)
- Large Vision-Language Model
  


## Documentation
- We provide a set of wrappers for FPS control, population randomization, etc.
Please refer to the [wrapper](doc/wrapper.md) for more details.
- We provide a document for adding a new environment in unrealzoo-gym. Please refer to the [addEnv](doc/addEnv.md) for more details.

# 🗓️ TODO List
- [ ]  Release an all-in-one package of the collected environments
- [ ]  Add gym interface for heterogeneous mutli-agent co-operation.
- [ ]  Expand the list of supported interactive actions.
- [ ]  Add more detailed examples for reinforcement learning agents.
- [ ]  Add more detailed examples for large vision-language models.

## License and Acknowledgments
The UnrealZoo project is licensed under the Apache 2.0. 
We acknowledge the following projects for their contributions to the UnrealZoo project:
- [UnrealCV](https://unrealcv.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [Unreal Engine](https://www.unrealengine.com/)
- [The fantastic Plugins/Content from UE Marketplace](https://www.unrealengine.com/marketplace)
  -  [Smart Locomotion](https://www.fab.com/zh-cn/listings/7f881534-bf40-493b-97b4-a917daa87af0)
  -  [Animal Pack](https://www.fab.com/zh-cn/listings/856c42d7-58a3-4b95-8f70-1302e5bdafa0)
  -  [Drivable Car](https://www.fab.com/zh-cn/listings/65a0844c-6be4-4e38-9d7a-b9697681a274)
  - ...

## Citation

If you use UnrealZoo in your research, please consider citing:

```bibtex
@software{UnrealZoo,
  author = {},
  title = {},
  month = {December},
  year = {2024},
  url = {}
}
