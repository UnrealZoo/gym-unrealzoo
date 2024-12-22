import argparse
import gym
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE
from pynput import keyboard
import time
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return 2
        return self.action_space.sample()


key_state = {
    'w': False,
    'a': False,
    's': False,
    'd': False,
    'space': False,
    '1': False,
    '2': False,
    'head_up': False,
    'head_down': False
}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False

def get_key_action():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['w']:
        action[0][1] = 100
    if key_state['s']:
        action[0][1] = -100
    if key_state['a']:
        action[0][0] = -30
    if key_state['d']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['1']:
        action[2] = 3
    if key_state['2']:
        action[2] = 4
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-SuburbNeighborhood_Day-MixedColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true',
                        help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=(640, 480))
    agent = RandomAgent(env.action_space[0])
    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs = env.reset()
    t0 = time.time()
    while True:
        action = get_key_action()
        # action = agent.act(obs[0])
        # action = ([0, 0], 0, 0)
        obs, rewards, done, info = env.step([action])
        # cv2.imshow('obs',obs[0])
        # cv2.waitKey(1)
        count_step += 1
        print(action)
        print(count_step)
        if count_step > 99:
            fps = count_step / (time.time() - t0)
            print('Failed')
            print('Fps:' + str(fps))
            break
        if done:
            fps = count_step / (time.time() - t0)
            print('Success')
            print('Fps:' + str(fps))
            break