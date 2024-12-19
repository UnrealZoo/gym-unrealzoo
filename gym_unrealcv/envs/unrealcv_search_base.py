import os
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.navigation import reward, reset_point
from gym_unrealcv.envs.navigation.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.navigation.interaction import Navigation
from gym_unrealcv.envs.agent.character import Character_API

'''
It is a general env for searching target object.

State : raw color image and depth (640x480) 
Action:  (linear velocity ,angle velocity , trigger) 
Done : Collision or get target place or False trigger three times.
Task: Learn to avoid obstacle and search for a target object in a room, 
      you can select the target name according to the Recommend object list in setting files
      
'''


class UnrealCvSearch_base(gym.Env):
    def __init__(self,
                 setting_file,
                 category,
                 reset_type='waypoint',  # testpoint, waypoint,
                 augment_env=None,  # texture, target, light
                 action_type='Discrete',  # 'Discrete', 'Continuous'
                 observation_type='Rgbd',  # 'color', 'depth', 'rgbd'
                 reward_type='bbox',  # distance, bbox, bbox_distance,
                 docker=False,
                 resolution=(160, 160),

                 ):

        setting = misc.load_env_setting(setting_file)
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets'][category]
        self.trigger_th = setting['trigger_th']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.player = setting['player']
        self.docker = docker
        self.reset_type = reset_type
        self.augment_env = augment_env

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution, offscreen=True)


        # connect UnrealCV
        self.unrealcv = Navigation(
                                # cam_id=self.cam_id,
                                   port=env_port,
                                   ip=env_ip,
                                   # targets=self.target_list,
                                   # env=self.unreal.path2env,
                                   resolution=resolution)
        # self.unrealcv = Character_API(port=env_port, ip=env_ip, resolution=resolution, comm_mode='tcp')
        # self.unrealcv.set_cam_illumination(0,'None')
        # self.unrealcv.set_cam_reflection(0,'None')

        self.unrealcv.pitch = self.pitch

        #  define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd' or self.observation_type == 'Mask'
        self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')

        # define reward type
        # distance, bbox, bbox_distance,
        # self.reward_type = reward_type
        self.reward_type = 'distance'

        self.reward_function = reward.Reward(setting)

        # set start position
        self.trigger_count = 0
        current_pose = self.unrealcv.get_pose(self.cam_id)

        # Only for virtual camera setting
        # current_pose[2] = self.height
        # self.unrealcv.set_location(self.cam_id, current_pose[:3])

        self.count_steps = 0

        self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)

        # for reset point generation and selection
        self.reset_module = reset_point.ResetPoint(setting, reset_type, current_pose)

    def step(self, action ):
        info = dict(
            Collision=False,
            Done=False,
            Trigger=0.0,
            Reward=0.0,
            Action=action,
            Bbox=[],
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Target=[],
            Direction=None,
            Waypoints=self.reset_module.waypoints,
            Color=None,
            Depth=None,
            Success=False
        )
        if action is not None:
            action = np.squeeze(action)
            if self.action_type == 'Discrete' :
                # (velocity, angle, info['Trigger']) = self.discrete_actions[action]
                (angle,velocity,jump,crouch) = self.discrete_actions[action]
            else:
                (velocity, angle, info['Trigger']) = action
            self.count_steps += 1
            info['Done'] = False

            # take action
            self.unrealcv.set_move_bp(self.player, [angle, velocity] )
            if jump==1:
                self.unrealcv.set_jump(self.player)
                jump_puni=-0.01
            else:
                jump_puni=0.0
            if crouch==1:
                self.unrealcv.set_crouch(self.player)
                crouch_puni = -0.01
            else:
                crouch_puni=0.0
        # info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)
        if self.unrealcv.get_hit(self.player)==0:
            info['Collision']=0
        else:
            info['Collision'] += 1
        # info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'soft') #for virtual camera
        info['Pose'] = self.unrealcv.get_obj_pose(self.player)

        #calculate relative pose
        pose_obs,relative_pose_2d= self.unrealcv.get_pose_states([info['Pose'],self.targets_pos[self.target_list[0]]])
        info['relative_pose']=np.array([relative_pose_2d[0][1][0],relative_pose_2d[0][1][1],self.targets_pos[self.target_list[0]][2]-info['Pose'][2]]) # distance,direction,height : point to player
        # the robot think that it found the target object,the episode is done
        # and get a reward by bounding box size
        # only three times false trigger allowed in every episode
        # if info['Trigger'] > self.trigger_th:
        #     self.trigger_count += 1
        #     # get reward
        #     if 'bbox' in self.reward_type:
        #         object_mask = self.unrealcv.read_image(self.cam_id, 'object_mask')
        #         boxes = self.unrealcv.get_bboxes(object_mask, self.target_list)
        #         info['Reward'], info['Bbox'] = self.reward_function.reward_bbox(boxes)
        #     else:
        #         info['Reward'] = 0
        #
        #     if info['Reward'] > 0 or self.trigger_count > 3:
        #         info['Done'] = True
        #         if info['Reward'] > 0 and self.reset_type == 'waypoint':
        #             self.reset_module.success_waypoint(self.count_steps)
        # else:
            # get reward
        distance, self.target_id = self.select_target_by_distance(info['Pose'][:3], self.targets_pos)
        info['Target'] = self.targets_pos[self.target_id]
        info['Direction'] = misc.get_direction(info['Pose'], self.targets_pos[self.target_id])

        # calculate reward according to the distance to target object
        if 'distance' in self.reward_type:
            # info['Reward'] = self.reward_function.reward_distance(distance)
            relative_oir_norm = np.fabs(info['Direction']) / 90.0
            reward_norm=np.tanh(self.reward_function.reward_distance(distance) - relative_oir_norm)
            info['Reward'] = reward_norm
        else:
            info['Reward'] = 0

        # if collision detected, the episode is done and reward is -1
        if info['Collision']>10 or info['Pose'][2]<self.height:
            info['Reward'] = -1
            info['Done'] = True
            if self.reset_type == 'waypoint':
                self.reset_module.update_dis2collision(info['Pose'])
        if distance<300 and np.fabs(info['Direction'])<10 :
            info['Success']=True
            info['Done'] = True
            info['Reward'] = 100
        # update observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

        # save the trajectory
        self.trajectory.append(info['Pose'][:6])
        info['Trajectory'] = self.trajectory
        if info['Done'] and len(self.trajectory) > 5 and self.reset_type == 'waypoint':
            self.reset_module.update_waypoint(info['Trajectory'])

        return state, info['Reward'], info['Done'], info

    def reset(self, ):
        # double check the resetpoint, it is necessary for random reset type
        self.unrealcv.set_obj_color(self.target_list[0], (255, 255, 255))

        collision = True
        while collision:
            current_pose = self.reset_module.select_resetpoint()
            # self.unrealcv.set_pose(self.cam_id, current_pose)
            self.unrealcv.set_obj_location(self.player, current_pose[:3])
            self.unrealcv.set_obj_rotation(self.player,current_pose[3:])
            # collision = self.unrealcv.set_move_bp(self.player, [0, 100])
            collision = self.unrealcv.get_hit(self.player)
        # self.unrealcv.set_pose(self.cam_id, current_pose)

        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.trigger_count = 0
        self.count_steps = 0
        self.reward_function.dis2target_initial, self.targetID_last = \
            self.select_target_by_distance(current_pose, self.targets_pos)

        return state

    def seed(self, seed=None):
        return seed

    def render(self, mode='rgb_array', close=False):
        if close:
            self.unreal.close()
        return self.unrealcv.img_color

    def close(self):
        self.unrealcv.client.disconnect()
        self.unreal.close()

    def get_action_size(self):
        return len(self.action)

    def select_target_by_distance(self, current_pos, targets_pos):
        # find the nearest target, return distance and targetid
        target_id = list(self.targets_pos.keys())[0]
        # distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 2)
        distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 3)

        for key, target_pos in targets_pos.items():
            # distance = self.unrealcv.get_distance(target_pos, current_pos, 2)
            distance = self.unrealcv.get_distance(target_pos, current_pos, 3)
            if distance < distance_min:
                target_id = key
                distance_min = distance
        return distance_min, target_id

