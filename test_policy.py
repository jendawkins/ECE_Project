import gym
import numpy as np
#import matplotlib.pyplot as plt
# __author__ = 'sudeep raja'
import numpy as np
# import _pickle as cPickle
import pickle
import heapq
from sklearn.neighbors import BallTree,KDTree
from neural_net import *
import time
import scipy
from scipy.spatial.distance import cdist
import operator
from neural_net import *

use_net = False
rng = np.random.RandomState(123456)
env = gym.make('MsPacman-v0')
if use_net:
    nnet = neural_net(env.observation_space.shape[0],env.action_space.n,1)
    nnet = torch.load('model.pt')
else:
    file_table = open('pacman3.pkl','rb')
    qec_table = pickle.load(file_table)
    file_table.close()


VISUALIZE = True
logdir = 'pacman3/'
images=True
if VISUALIZE:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    env = gym.wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: 1)

def predict_net(new,net):
    st,act = new
    if np.isscalar(st):
        st = [st]
    s_in = torch.Tensor([st])
    a_in = torch.Tensor([[act]])
    net.eval()
    pred = net(s_in, a_in)
    return pred.detach().numpy()[0][0]

def predict_table(new,table,knn=11):
    if len(table)==0:
        return 0.0
    if new in table.keys():
        return table[new]
    states,actions = zip(*[key for key,item in table.items() if key[1]==new[1]])
    if len(states) < knn:
        k = len(states)
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
    dist, ind = tree.query(query_pt, k)
    value = 0
    for index in ind[0]:
        value += table[(states[index],actions[index])]
    return value / knn

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


for episodes in range(20):
    if VISUALIZE:
        env.render()
        time.sleep(0.05)
    # self.env.reset()
    # state = self.env.observation_space.sample()
    # self.env.env.state = state
    state = env.reset()
    done = False
    # epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*i)
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

        for action in range(env.action_space.n):
            if use_net:
                vt = predict_net((state,action),nnet)
            else:
                vt = predict_table((state_t,action),qec_table)
            value_t.append(vt)
        if len(set(value_t))==1:
            maximum_action = random.randint(0, env.action_space.n)
        else:
            maximum_action = np.argmax(value_t)
            # if i==2:
                # import pdb; pdb.set_trace()
        next_state, reward, done , _ = env.step(maximum_action)

        state = next_state
        ep_reward += reward # total reward for this episode: 1 if convergence
        steps += 1.0
