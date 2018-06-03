import gym
import numpy as np
import matplotlib.pyplot as plt

# __author__ = 'sudeep raja'
import numpy as np
# import _pickle as cPickle
import pickle
import heapq
from sklearn.neighbors import BallTree,KDTree


def knn(table,new,knn=11):
    if len(table)==0:
        return 0.0
    if new in table.keys():
        return table[new]
    states,actions = zip(*[key for key,item in table.items() if key[1]==new[1]])
    if len(table) < knn:
        k = len(table)
    else:
        k = knn
    if np.isscalar(states[0]):
        dim2 = 1
        query_pt= [[new[0]]]
    else:
        dim2 = len(states[0])
        query_pt = np.array(new[0]).reshape(1,-1)
    states_a = np.reshape(states,(len(states),dim2))
    tree = KDTree(states_a)
    # import pdb; pdb.set_trace()
    dist, ind = tree.query(query_pt, k)
    value = 0
    for index in ind[0]:
        value += table[(states[index],actions[index])]

    return value / knn

def update_table(table,R,new):
    if new in table.keys():
        if R>table[new]:
            table[new] = R
    else:
        table[new] = R
    return table

# Create the CartPole game environment
# env = gym.make('FrozenLake-v0')
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )
# env = gym.make('FrozenLakeNotSlippery-v0')
continuous = True
env = gym.make('MountainCar-v0')
env.reset()

rng = np.random.RandomState(123456)
# obs_dim = 84*84
action_size = env.action_space.n
# state_size = env.observation_space.shape[0]
if continuous:
    state_dimension = env.observation_space.shape[0]
    state_size = env.observation_space.shape[0]
else:
    state_dimension = 1
    state_size = env.observation_space.n
buffer_size = 100000
ec_discount = .99
min_epsilon = 0.01
decay_rate = 10000
qec_table = {}
# qec_table = QECTable(action_size,rng,obs_dim,state_dimension,buffer_size,images=False)
ep_avg_reward = []
total_reward = []
total_sum_reward = 0
epochs = 5000
for i in range(epochs):
    epoch_steps = 0
    episodes_per_epoch = 0
    reward_per_epoch = 0
    while epoch_steps < 10000:
        state = env.reset()
        done = False
        epsilon = min_epsilon + (1.0 - min_epsilon)*np.exp(-decay_rate*i)
        steps = 0.
        ep_reward = 0.
        trace_list = []
        while not done:
            value_t = []
            if not np.isscalar(state):
                state = tuple(state)
            # epsilon greedy
            if rng.rand() < epsilon:
                maximum_action = rng.randint(0, action_size)
            else:
                for action in range(action_size):
                    value_t.append(knn(qec_table,(state,action)))
                    if sum(value_t)==0:
                        maximum_action = rng.randint(0, action_size)
                    else:
                        maximum_action = np.argmax(value_t)

            next_state, reward, done , _ = env.step(maximum_action)

            trace_list.append((state, maximum_action, reward, done))
            state = next_state
            ep_reward += reward # total reward for this episode: 1 if convergence
            steps += 1.0
        # import pdb; pdb.set_trace()
        reward_per_epoch += ep_reward
        epoch_steps += steps
        episodes_per_epoch += 1
        # ep_avg_reward.append(ep_reward)
        # total_reward.append(ep_reward)
        q_return = 0.
        for j in range(len(trace_list)-1, -1, -1):
            node = trace_list[j]
            q_return = q_return * ec_discount + node[2]
            qec_table = update_table(qec_table,q_return,(node[0],node[1]))
            # if node[2]==1:
                # import pdb; pdb.set_trace()
        # qec_table.update(node[0], node[1], q_return)
    # import pdb; pdb.set_trace()
    total_reward.append(reward_per_epoch/episodes_per_epoch)
    total_sum_reward += reward_per_epoch
    # print('Average Reward: '+ str(sum(ep_avg_reward)/len(ep_avg_reward)))
    print('Average Epoch ' + str(i) + ' Reward: ' + str(total_reward[-1]))
    print('Total Reward: ' + str(total_sum_reward))
