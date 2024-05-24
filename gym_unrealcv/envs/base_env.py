import os
import time
import warnings

import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.utils import env_unreal, misc
from unrealcv.api import UnrealCv_API
from unrealcv.launcher import RunUnreal
from gym_unrealcv.envs.tracking.interaction import Tracking
import gym_unrealcv
import cv2
import random
import sys
''' 
It is a base env for general purpose agent-env interaction, including single/multi-agent navigation, tracking, etc.
Observation : raw color image and depth
Action:  Discrete/Continuous
Done : define by the task wrapper
'''

# TODO: agent apis for blueprints commands
# TODO: config env by parapmeters
# TODO: matain a general agent list
class UnrealCv_base(gym.Env):
    def __init__(self,
                 setting_file, # the setting file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(240, 240)
                 ):

        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.max_steps = setting['max_steps']
        self.height = setting['height']
        self.cam_id = [setting['third_cam']['cam_id']]
        self.agent_configs = setting['agents']
        self.env_configs = setting["env"]
        self.agents = misc.convert_dict(self.agent_configs)

        # TODO: it is useless.
        self.character = {
            'player': [], # the list of player to control
            'npc': [], # the list of Non-player character
            'freeze': [], # the list of player that exists in the scene, but it is frozen
        }

        self.height_top_view = setting['third_cam']['height_top_view']

        self.objects_list = self.env_configs["objects"]
        self.reset_area = setting['reset_area']
        self.max_player_num = setting['max_player_num']  # the max players number

        self.safe_start = setting['safe_start']
        self.interval = setting['interval']
        self.random_init = setting['random_init']
        self.start_area = self.get_start_area(self.safe_start[0], 500) # the start area of the agent, where we don't put obstacles

        self.count_eps = 0
        self.count_steps = 0

        # env configs
        self.docker = False
        self.resolution = resolution
        self.display = None
        self.use_opengl = False
        self.offscreen_rendering = False
        self.nullrhi = False
        self.launched = False

        self.agents_category = ['player'] # the agent category we use in the env
        self.protagonist_id = 0

        # init agents
        self.player_list = list(self.agents.keys())
        self.cam_list = [self.agents[player]['cam_id'] for player in self.player_list]

        # define action space
        self.action_type = action_type
        assert self.action_type in ['Discrete', 'Continuous']
        self.action_space = [self.define_action_space(self.action_type, self.agents[obj]) for obj in self.player_list]

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose']
        self.observation_space = [self.define_observation_space(self.cam_list[i], self.observation_type, resolution)
                                  for i in range(len(self.player_list))]

        # config unreal env
        if 'linux' in sys.platform:
            env_bin = setting['env_bin']
        elif 'win' in sys.platform:
            env_bin = setting['env_bin_win']
        if 'env_map' in setting.keys():
            env_map = setting['env_map']
        else:
            env_map = None

        self.ue_binary = RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)

    def step(self, actions):
        info = dict(
            Collision=0,
            Done=False,
            Reward=0.0,
            Action=actions,
            Pose=[],
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
            Relative_Pose=[]
        )

        actions2player = self.action_mapping(actions, self.player_list)

        move_cmds = [self.unrealcv.set_move_bp(obj, actions2player[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2player[i] is not None]
        self.unrealcv.batch_cmd(move_cmds, None)
        # self.unrealcv.set_move_batch(self.player_list, actions2player)
        self.count_steps += 1

        # get states
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        self.obj_poses = obj_poses
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)

        self.img_show = self.prepare_img2show(self.protagonist_id, observations)

        pose_obs, relative_pose = self.get_pose_states(obj_poses)

        # prepare the info
        info['Pose'] = obj_poses[self.protagonist_id]
        info['Relative_Pose'] = relative_pose
        info['Pose_Obs'] = pose_obs
        info['Reward'] = np.zeros(len(self.player_list))

        if self.count_steps > self.max_steps:
            info['Done'] = True
        return observations, info['Reward'], info['Done'], info

    def reset(self, ):
        if not self.launched:  # first time to launch
            self.launched = self.launch_ue_env()
            self.init_agents()
            self.init_objects()

        self.count_close = 0
        self.count_steps = 0
        self.count_eps += 1

        # stop move and disable physics
        for i, obj in enumerate(self.player_list):
            if self.agents[obj]['agent_type'] in self.agents_category:
                if not self.agents[obj]['internal_nav']:
                    self.unrealcv.set_move_bp(obj, [0, 0])
                    self.unrealcv.set_phy(obj, 0)
            elif self.agents[obj]['agent_type'] == 'drone':
                self.unrealcv.set_move_bp(obj, [0, 0, 0, 0])
                self.unrealcv.set_phy(obj, 0)

        # reset target location
        for obj in self.player_list:
            self.unrealcv.set_obj_location(obj, self.sample_init_pose(self.random_init))


        # set view point
        for obj in self.player_list:
            self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], self.agents[obj]['relative_rotation'])
            self.unrealcv.set_phy(obj, 1) # enable physics

        # get state
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        self.obj_poses = obj_poses
        self.img_show = self.prepare_img2show(self.protagonist_id, observations)

        return observations

    def close(self):
        if self.launched:
            self.unrealcv.client.disconnect()
            self.ue_binary.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.ue_binary.close()
        return self.img_show

    def seed(self, seed=None):
        np.random.seed(seed)
        # if seed is not None:
        #     self.player_num = seed % (self.max_player_num-2) + 2

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_topview(self, current_pose, cam_id):
        cam_loc = current_pose[:3]
        cam_loc[-1] = self.height_top_view
        cam_rot = [0, 0, -90]
        self.unrealcv.set_location(cam_id, cam_loc)
        self.unrealcv.set_rotation(cam_id, cam_rot)

    def get_relative(self, pose0, pose1):  # pose0-centric
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.unrealcv.get_distance(pose1, pose0, 2)
        # distance_norm = distance / self.exp_distance
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance]
        return obs_vector, distance, angle

    def prepare_observation(self, observation_type, img_list, mask_list, depth_list, pose_list):
        if observation_type == 'Depth':
            return np.array(depth_list)
        elif observation_type == 'Mask':
            return np.array(mask_list)
        elif observation_type == 'Color':
            return np.array(img_list)
        elif observation_type == 'Rgbd':
            return np.append(np.array(img_list), np.array(depth_list), axis=-1)
        elif observation_type == 'Pose':
            return np.array(pose_list)

    def rotate2exp(self, yaw_exp, obj, th=1):
        yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
        delta_yaw = yaw_exp - yaw_pre
        while abs(delta_yaw) > th:
            if 'Drone' in obj:
                self.unrealcv.set_move_bp(obj, [0, 0, 0, np.clip(delta_yaw, -60, 60)/60*np.pi])
            else:
                self.unrealcv.set_move_bp(obj, [np.clip(delta_yaw, -60, 60), 0])
            yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
            delta_yaw = (yaw_exp - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        return delta_yaw

    def relative_metrics(self, relative_pose):
        # compute the relative relation (collision, in-the-view, misleading) among agents for rewards and evaluation metrics
        info = dict()
        relative_dis = relative_pose[:, :, 0]
        relative_ori = relative_pose[:, :, 1]
        collision_mat = np.zeros_like(relative_dis)
        collision_mat[np.where(relative_dis < 100)] = 1
        collision_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['collision'] = collision_mat
        info['dis_ave'] = relative_dis.mean() # average distance among players, regard as a kind of density metric

        return info

    def add_agent(self, name, loc, refer_agent):
        # print(f'add {name}')
        new_dict = refer_agent.copy()
        cam_num = self.unrealcv.get_camera_num()
        self.unrealcv.new_obj(refer_agent['class_name'], name, random.sample(self.safe_start, 1)[0])
        self.player_list.append(name)
        if self.unrealcv.get_camera_num() > cam_num:
            new_dict['cam_id'] = cam_num
        else:
            new_dict['cam_id'] = -1
        self.cam_list.append(new_dict['cam_id'])
        self.unrealcv.set_obj_scale(name, refer_agent['scale'])
        self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
        self.unrealcv.set_random(name, 0)
        self.unrealcv.set_interval(self.interval, name)
        self.unrealcv.set_obj_location(name, loc)
        self.action_space.append(self.define_action_space(self.action_type, agent_info=new_dict))
        self.observation_space.append(self.define_observation_space(new_dict['cam_id'], self.observation_type, self.resolution))
        # self.unrealcv.set_phy(name, 0)
        return new_dict

    def remove_agent(self, name, freeze=False):
        # print(f'remove {name}')
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        if freeze:
            self.freeze_list.append(name)  # the agent still exists in the scene, but it is frozen
        else:
            self.unrealcv.destroy_obj(name)  # the agent is removed from the scene
            self.agents.pop(name)

    def remove_cam(self, name):
        cam_id = self.agents[name]['cam_id']
        cam_list = []
        for obj in self.player_list:
            if self.agents[obj]['cam_id'] > cam_id and cam_id > 0:
                self.agents[obj]['cam_id'] -= 1
            cam_list.append(self.agents[obj]['cam_id'])
        return cam_list

    def define_action_space(self, action_type, agent_info):
        if action_type == 'Discrete':
            return spaces.Discrete(len(agent_info["discrete_action"]))
        elif action_type == 'Continuous':
            return spaces.Box(low=np.array(agent_info["continuous_action"]['low']),
                              high=np.array(agent_info["continuous_action"]['high']), dtype=np.float32)

    def define_observation_space(self, cam_id, observation_type, resolution=(160, 120)):
        if observation_type == 'Pose' or cam_id < 0:
            observation_space = spaces.Box(low=-100, high=100, shape=(6,),
                                               dtype=np.float16)  # TODO check the range and shape
        else:
            if observation_type == 'Color' or observation_type == 'CG' or observation_type == 'Mask':
                img_shape = (resolution[1], resolution[0], 3)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
            elif observation_type == 'Depth':
                img_shape = (resolution[1], resolution[0], 1)
                observation_space = spaces.Box(low=0, high=100, shape=img_shape, dtype=np.float16)
            elif observation_type == 'Rgbd':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)

        return observation_space

    def sample_init_pose(self, use_reset_area=False, num_agents=1):
        # sample poses to reset the agents
        if num_agents < len(self.safe_start):
            use_reset_area = True
            warnings.warn('The number of agents is less than the number of pre-defined start points, random sample points from the pre-defined area instead.')
        if use_reset_area:
            locations, _ = self.unrealcv.get_startpoint(reset_area=self.reset_area, exp_height=self.height)
        else:
            locations = random.choice(self.safe_start, num_agents) # sample one pre-defined start point
        # self.unrealcv.set_obj_location(self.player_list[self.target_id], target_pos)

        return locations


    def environment_augmentation(self, player_mesh=False, player_texture=False,
                                 light=False, background_texture=False,
                                 layout=False, layout_texture=False):
        if player_mesh:  # random human mesh
            for obj in self.player_list:
                if self.agents[obj]['agent_type'] == 'player':
                    if self.env_name == 'MPRoom':
                        map_id = [2, 3, 6, 7, 9]
                        spline = False
                        app_id = np.random.choice(map_id)
                    else:
                        map_id = [1, 2, 3, 4]
                        spline = True
                        app_id = np.random.choice(map_id)
                    self.unrealcv.set_appearance(obj, app_id, spline)
                if self.agents[obj]['agent_type'] == 'animal':
                    map_id = [2, 5, 6, 7, 11, 12, 16]
                    spline = True
                    app_id = np.random.choice(map_id)
                    self.unrealcv.set_appearance(obj, app_id, spline)
        # random light and texture of the agents
        if player_texture:
            if self.env_name == 'MPRoom':  # random target texture
                for obj in self.player_list:
                    if self.agents[obj]['agent_type'] == 'player':
                        self.unrealcv.random_player_texture(obj, self.textures_list, 3)
        if light:
            self.unrealcv.random_lit(self.env_configs["lights"])

        # random the texture of the background
        if background_texture:
            self.unrealcv.random_texture(self.env_configs["backgrounds"], self.textures_list, 3)

        # random place the obstacle
        if layout:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
                                           15, self.reset_area, self.start_area, layout_texture)

    def get_pose_states(self, obj_pos):
        # get the relative pose of each agent and the absolute location and orientation of the agent
        pose_obs = []
        player_num = len(obj_pos)
        np.zeros((player_num, player_num, 2))
        relative_pose = np.zeros((player_num, player_num, 2))
        for j in range(player_num):
            vectors = []
            for i in range(player_num):
                obs, distance, direction = self.get_relative(obj_pos[j], obj_pos[i])
                yaw = obj_pos[j][4]/180*np.pi
                # rescale the absolute location and orientation
                abs_loc = [obj_pos[i][0], obj_pos[i][1],
                           obj_pos[i][2], np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)

        return np.array(pose_obs), relative_pose

    def launch_ue_env(self, comm_mode='tcp', **kwargs):
        # launch the UE4 binary and connect to UnrealCV
        env_ip, env_port = self.ue_binary.start(**kwargs)
        # connect UnrealCV
        self.unrealcv = Tracking(port=env_port, ip=env_ip, resolution=self.resolution, comm_mode=comm_mode)

        return True

    def init_agents(self):
        for obj in self.player_list.copy(): # the agent will be fully removed in self.agents
            if self.agents[obj]['agent_type'] not in self.agents_category:
                self.remove_agent(obj)

        for obj in self.player_list:
            self.agents[obj]['scale'] = self.unrealcv.get_obj_scale(obj)
            self.unrealcv.set_random(obj, 0)
            self.unrealcv.set_interval(self.interval, obj)

        self.unrealcv.build_color_dict(self.player_list)
        self.cam_flag = self.unrealcv.get_cam_flag(self.observation_type)

    def init_objects(self):
        self.unrealcv.init_objects(self.objects_list)

    def prepare_img2show(self, index, states):
        if self.observation_type == 'Rgbd':
            return states[index][:, :, :3]
        elif self.observation_type in ['Color', 'Depth', 'Gray', 'CG', 'Mask']:
            return states[index]
        else:
            return None

    def set_population(self, num_agents):
        while len(self.player_list) < num_agents:
            refer_agent = self.agents[random.choice(list(self.agents.keys()))]
            name = f'{refer_agent["agent_type"]}_EP{self.count_eps}_{len(self.player_list)}'
            self.agents[name] = self.add_agent(name, random.choice(self.safe_start), refer_agent)
        while len(self.player_list) > num_agents:
            self.remove_agent(self.player_list[-1])  # remove the last one

    def set_npc(self):
        # TODO: set the NPC agent
        return self.player_list.index(random.choice([x for x in self.player_list if x > 0]))

    def set_agent(self):
        # the agent is controlled by the external controller
        return self.cam_list.index(random.choice([x for x in self.cam_list if x > 0]))

    def check_visibility(self, cam_id):
        mask = self.unrealcv.read_image(self.cam_id[cam_id], 'object_mask', 'fast')
        mask, bbox = self.unrealcv.get_bbox(mask, self.player_list[self.target_id], normalize=False)
        mask_percent = mask.sum()/(self.resolution[0] * self.resolution[1])
        return mask_percent

    def action_mapping(self, actions, player_list):
        actions2player = []
        for i, obj in enumerate(player_list):
            if actions[i] is None:  # if the action is None, then we don't control this agent
                actions2player.append(None)  # place holder
                continue
            if self.action_type == 'Discrete':
                act_index = actions[i]
                act_now = self.agents[obj]["discrete_action"][act_index]
                actions2player.append(act_now)
            else:
                actions2player.append(actions[i])
        return actions2player
