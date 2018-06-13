# load pckle file
import pickle
import numpy as np
import gym
import scipy
import heapq
from PIL import Image
import random
from scipy.spatial.distance import cdist


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_qstar(episodes,images,gamma,env):
    q_value = {}
    for episodes in range(episodes):
        env.reset()
        state = env.observation_space.sample()
        env.env.state = state
        done = False
        steps = 0.
        ep_reward = 0.
        trace_list = []
        while not done:
            value_t = []
            if images:
                state = rgb2gray(state)
                state = scipy.misc.imresize(state, size=(84,84))
                matrix_projection = rng.randn(64,84*84).astype(np.float32)
                state = np.dot(matrix_projection, state.flatten())
            if not np.isscalar(state):
                state_t = tuple(state)
            else:
                state_t = state
            maximum_action = random.randint(0, env.action_space.n-1)
            ns, reward, done , _ = env.step(maximum_action)
            # import pdb; pdb.set_trace()
            try:
                q_ns = [value for key, value in q_value.items() if key[0] == tuple(ns)]
            except:
                import pdb; pdb.set_trace()
            if len(q_ns)>0 and not done:
                q_value[(state_t, maximum_action )] = reward + gamma * np.max(q_ns)
            else:
                q_value[(state_t, maximum_action )] = reward
            # trace_list.append((state, maximum_action, reward, done))
            state = ns
            ep_reward += reward # total reward for this episode: 1 if convergence
            steps += 1.0
    return q_value
# states,actions = zip(*[key for key,item in self.qec_table.items()])
# goals = [item for key, item in self.qec_table.items()]

def get_filtered(delta_factor, const, table):
    sa, goals = zip(*list(table.items()))
    states,actions = zip(*(sa))
    delta = np.std(np.array(states),axis = 0)
    delta = np.sum(np.sqrt(np.power(delta,2)))
    # delt = np.sqrt(delta[0]**2 + delta[1]**2)/delta_factor

    for i, new in enumerate(sa):
        if new in table.keys():
            # sample_dict = removekey(table, new)
            idxs = np.where(cdist(np.array(states).reshape(len(states),-1),np.array([[new[0]]]))<delta)[0]
            med_arr = np.median(np.array(goals)[idxs])
            for idx in idxs:
                if (states[idx],actions[idx]) in table.keys():
                    if goals[idx] + const < med_arr:
                        table.pop((states[idx],actions[idx]))
    return table

env = gym.make('MountainCar-v0')
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )
# env = gym.make('FrozenLakeNotSlippery-v0')
const = 0
file_table = open('mt_car_wModel_noFilt.pkl','rb')
qec_table = pickle.load(file_table)
file_table.close()
images = False
gamma = .1
qstar = get_qstar(1000,images,gamma,env)

qhat = qec_table
delta_factor = 1
qhat_prime = get_filtered(delta_factor, const, qec_table)
import pdb; pdb.set_trace()
tot = 0
for key in qstar.keys():
    if key in qhat.keys() and key in qhat_prime.keys():
        diff1 = abs(qstar[key] - qhat_prime[key])
        diff2 = abs(qstar[key] - qhat[key])
        tot += diff1 - diff2
