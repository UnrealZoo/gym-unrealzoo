from gym.envs.registration import register
import logging
from gym_unrealcv.envs.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker

# Searching/Navigation
# -------------------------------------------------
for env in ['RealisticRoom', 'Arch1','Demo_Roof','IndustrialArea']:
    setting_file = 'searching/{env}.json'.format(env=env)
    settings = load_env_setting(setting_file)
    for i, reset in enumerate(['random', 'waypoint', 'testpoint']):
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd','Mask']:  # observation type
                for category in settings['targets']:
                    register(
                        id='UnrealSearch-{env}{category}-{action}{obs}-v{reset}'.format(env=env, category=category, action=action, obs=obs, reset=i),
                        entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
                        kwargs={'setting_file': 'searching/{env}.json'.format(env=env),
                                'category': category,
                                'reset_type': reset,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'bbox_distance',  # bbox, distance, bbox_distance
                                'docker': use_docker,
                                },
                        max_episode_steps=1000
                    )

# ------------------------------------------------------------------
# Robot Arm
# "CRAVES: Controlling Robotic Arm With a Vision-Based Economic System", CVPR 2019
for action in ['Discrete', 'Continuous']:  # action type
    for obs in ['Pose', 'Color', 'Depth', 'Rgbd']:
        for i in range(3):
            register(
                    id='UnrealArm-{action}{obs}-v{version}'.format(action=action, obs=obs, version=i),
                    entry_point='gym_unrealcv.envs:UnrealCvRobotArm_reach',
                    kwargs={'setting_file': 'robotarm/robotarm_reach.json',
                            'action_type': action,
                            'observation_type': obs,
                            'docker': use_docker,
                            'version': i
                            },
                    max_episode_steps=100
                        )

# -----------------------------------------------------------------------
# Tracking
# "End-to-end Active Object Tracking via Reinforcement Learning", ICML 2018
for env in ['City1', 'City2']:
    for target in ['Malcom', 'Stefani']:
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd']:  # observation type
                for path in ['Path1', 'Path2']:  # observation type
                    for i, reset in enumerate(['Static', 'Random']):
                        register(
                            id='UnrealTrack-{env}{target}{path}-'
                               '{action}{obs}-v{reset}'.format(env=env, target=target, path=path,
                                                               action=action, obs=obs, reset=i),
                            entry_point='gym_unrealcv.envs:UnrealCvTracking_spline',
                            kwargs={'setting_file': 'tracking/v0/{env}{target}{path}.json'.format(
                                env=env, target=target, path=path),
                                    'reset_type': reset,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reward_type': 'distance',
                                    'docker': use_docker,
                                    },
                            max_episode_steps=3000
                            )


# "Pose-Assisted Multi-Camera Collaboration for Active Object Tracking", AAAI 2020
for env in ['MCRoom', 'Garden', 'UrbanTree']:
    for i in range(7):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', 'None',
                            'RandomInterval', 'GoalInterval', 'InternalInterval', 'NoneInterval']:
                    name = 'Unreal{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    setting_file = 'tracking/multicam/{env}.json'.format(env=env)
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvMC',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )

for env in ['FlexibleRoom', 'Garden', 'UrbanTree']:
    for i in range(7):  # reset type
        for action in ['Discrete', 'Continuous']:  # action type
            for obs in ['Color', 'Depth', 'Rgbd', 'Gray']:  # observation type
                for nav in ['Random', 'Goal', 'Internal', 'None',
                            'RandomInterval', 'GoalInterval', 'InternalInterval']:
                    name = 'UnrealMC{env}-{action}{obs}{nav}-v{reset}'.format(env=env, action=action, obs=obs, nav=nav, reset=i)
                    setting_file = 'tracking/mcmt/{env}.json'.format(env=env)
                    register(
                        id=name,
                        entry_point='gym_unrealcv.envs:UnrealCvMultiCam',
                        kwargs={'setting_file': setting_file,
                                'reset_type': i,
                                'action_type': action,
                                'observation_type': obs,
                                'reward_type': 'distance',
                                'docker': use_docker,
                                'nav': nav
                                },
                        max_episode_steps=500
                    )

maps = ['Greek_Island', 'supermarket', 'Brass_Gardens', 'Brass_Palace', 'Brass_Streets',
            'EF_Gus', 'EF_Lewis_1', 'EF_Lewis_2', 'EF_Grounds', 'Eastern_Garden', 'Western_Garden', 'Colosseum_Desert',
            'Desert_ruins', 'SchoolGymDay', 'Venice', 'TrainStation', 'Stadium', 'IndustrialArea', 'ModularBuilding',
            'TemplePlaza', 'DowntownWest', 'TerrainDemo', 'InteriorDemo_NEW', 'AncientRuins', 'Grass_Hills', 'ChineseWaterTown_Ver1',
            'ContainerYard_Night', 'ContainerYard_Day', 'Old_Factory_01', 'racing_track', 'Watermills', 'WildWest',
            'SunsetMap', 'Hospital', 'Medieval_Castle', 'Real_Landscape', 'UndergroundParking', 'Demonstration_Castle',
            'Demonstration_Cave', 'Arctic', 'Medieval_Daytime', 'Medieval_Nighttime', 'ModularGothic_Day', 'ModularGothic_Night',
            'UltimateFarming', 'RuralAustralia_Example_01', 'RuralAustralia_Example_02', 'RuralAustralia_Example_03',
            'LV_Soul_Cave', 'Dungeon_Demo_00', 'SwimmingPool', 'DesertMap', 'RainMap', 'SnowMap', 'ModularVictorianCity_scene1',
            'SuburbNeighborhood_Day', 'SuburbNeighborhood_Night', 'Storagehouse', 'OceanFloor',
            'ModularNeighborhood', 'ModularSciFiVillage', 'ModularSciFiSeason1', 'LowPolyMedievalInterior_1',
            'QA_Holding_Cells_A', 'ParkingLot','track_train','Demo_Roof','demonstration_BUNKER','Map_ChemicalPlant_1'
            ]

Tasks = ['Rendezvous', 'Rescue', 'Track']
Observations = ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose','MaskDepth','ColorMask']
Actions = ['Discrete', 'Continuous', 'Mixed']
# Env for general purpose active object tracking
# Base env for general purpose multi-agent interaction
for env in maps:
    for i in range(7):  # reset type
        for action in Actions:  # action type
            for obs in Observations:  # observation type
                        name = 'UnrealAgent-{env}-{action}{obs}-v{reset}'.format(env=env, action=action, obs=obs, target=target, reset=i)
                        setting_file = 'env_config/{env}.json'.format(env=env)
                        register(
                            id=name,
                            entry_point='gym_unrealcv.envs:UnrealCv_base',
                            kwargs={'setting_file': setting_file,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reset_type': i,
                                    },
                            max_episode_steps=500
                            )
# Task-oriented envs
for env in maps:
    for i in range(7):  # reset type
        for action in Actions:  # action type
            for obs in Observations:  # observation type
                for task in Tasks:
                        name = f'Unreal{task}-{env}-{action}{obs}-v{i}'
                        setting_file = 'env_config/{env}.json'.format(env=env)
                        register(
                            id=name,
                            entry_point=f'gym_unrealcv.envs:{task}',
                            kwargs={'env_file': setting_file,
                                    'action_type': action,
                                    'observation_type': obs,
                                    'reset_type': i,
                                    },
                            max_episode_steps=2000
                            )