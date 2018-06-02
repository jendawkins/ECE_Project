import gym
import numpy as np
import matplotlib.pyplot as plt

# __author__ = 'sudeep raja'
import numpy as np
# import _pickle as cPickle
import pickle
import heapq
# import image_preprocessing as ip
from sklearn.neighbors import BallTree,KDTree


class LRU_KNN:
    def __init__(self, capacity,dimension_result):
        self.capacity = capacity
        self.states = np.zeros((capacity,dimension_result))
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None

    def peek(self,key,value,modify):
        if self.curr_capacity==0:
            return None
        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        if np.allclose(self.states[ind],key):
            self.lru[ind] = self.tm
            self.tm +=0.01
            if modify:
                self.q_values[ind] = max(self.q_values[ind],value)
            return self.q_values[ind]

        return None

    def knn_value(self, key, knn=11):
        if self.curr_capacity==0:
            return 0.0
        if self.curr_capacity < knn:
            k = self.curr_capacity
        else:
            k = knn
        dist, ind = self.tree.query([key], k)

        value = 0.0
        for index in ind[0]:
            value += self.q_values[index]
            self.lru[index] = self.tm
            self.tm+=0.01

        return value / knn

    def add(self, key, value):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values[old_index] = value
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.q_values[self.curr_capacity] = value
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity+=1
        self.tm += 0.01
        self.tree = KDTree(self.states[:self.curr_capacity])


class QECTable(object):
    def __init__(self, num_actions, rng, observation_dimension, state_dimension, buffer_size, images):
        # self.knn = knn
        self.images = images
        self.ec_buffer = []
        self.buffer_maximum_size = buffer_size
        self.rng = rng
        for i in range(num_actions):
            self.ec_buffer.append(LRU_KNN(buffer_size,state_dimension))

        # projection
        """
        I tried to make a function self.projection(state)
        but cPickle can not pickle an object that has an function attribute
        """
        if self.images:
            self._initialize_projection_function(state_dimension, observation_dimension)

    def _initialize_projection_function(self, dimension_result, dimension_observation, p_type='random'):
        if p_type == 'random':
            self.matrix_projection = self.rng.randn(dimension_result, dimension_observation).astype(np.float32)
        elif p_type == 'VAE':
            pass
        else:
            raise ValueError('unrecognized projection type')

    """estimate the value of Q_EC(s,a)  O(N*logK*D)  check existence: O(N) -> KNN: O(D*N*logK)"""
    def estimate(self, s, a):
        if self.images:
            state = np.dot(self.matrix_projection, s.flatten())
        else:
            state = s
            if np.isscalar(state):
                state = [s]
        q_value = self.ec_buffer[a].peek(state,None,modify = False)
        if q_value!=None:
            return q_value
        return self.ec_buffer[a].knn_value(state)

    def update(self, s, a, r):  # s is 84*84*3;  a is 0 to num_actions; r is reward
        if self.images:
            state = np.dot(self.matrix_projection, s.flatten())
        else:
            state = s
            if np.isscalar(state):
                state = [s]
        q_value = self.ec_buffer[a].peek(state,r,modify = True)
        if q_value==None:
            self.ec_buffer[a].add(state,r)

# Create the CartPole game environment
env = gym.make('FrozenLake-v0')
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
continuous = False
# env = gym.make('MountainCar-v0')
env.reset()

rng = np.random.RandomState(123456)
obs_dim = 84*84
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
decay_rate = 100
qec_table = QECTable(action_size,rng,obs_dim,state_dimension,buffer_size,images=False)
trace_list = []
for i in range(50000):
    state = env.reset()
    done = False
    epsilon = min_epsilon + (1.0 - min_epsilon)*np.exp(-decay_rate*i)
    while not done:
        value_t = []
        # epsilon greedy
        if rng.rand() < epsilon:
            maximum_action = rng.randint(0, action_size)
        else:
            for action in range(action_size):
                value_t.append(qec_table.estimate(state, action))
                if sum(value_t)==0:
                    maximum_action = rng.randint(0, action_size)
                else:
                    maximum_action = np.argmax(value_t)

        state, reward, done , _ = env.step(maximum_action)

        trace_list.append((state, maximum_action, reward, done))
    q_return = 0.
    for j in range(len(trace_list)-1, -1, -1):
        node = trace_list[j]
        q_return = q_return * ec_discount + node[2]
        qec_table.update(node[0], node[1], q_return)
    if not i % 100:
        print(q_return)
        print(epsilon)
