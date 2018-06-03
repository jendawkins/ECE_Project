import gym
import numpy as np
import matplotlib.pyplot as plt
# __author__ = 'sudeep raja'
import numpy as np
# import _pickle as cPickle
import pickle
import heapq
from sklearn.neighbors import BallTree,KDTree
from neural_net import *

class Episodic_Control():
    def __init__(self, net, environment, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn):

        self.env = environment
        self.rng = rng
        self.buffer_size = buffer_size
        self.ec_discount = ec_discount
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.knn = knn
        self.qec_table = {}
        self.action_size = env.action_space.n
        # state_size = env.observation_space.shape[0]
        if continuous:
            self.state_dimension = self.env.observation_space.shape[0]
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_dimension = 1
            self.state_size = self.env.observation_space.n

        self.net = neural_net(self.state_dimension,self.action_size,1)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(),lr = .01)

    def knn(self,new,knn):
        if len(self.qec_table)==0:
            return 0.0
        if new in self.qec_table.keys():
            return table[new]
        states,actions = zip(*[key for key,item in self.qec_table.items()])
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

    def update_table(self,R,new):
        if new in self.qec_table.keys():
            if R>table[new]:
                self.qec_table[new] = R
        else:
            self.qec_table[new] = R

    def train(self):
        ep_avg_reward = []
        self.total_reward = []
        self.total_sum_reward = 0
        for i in range(self.epochs):
            epoch_steps = 0
            episodes_per_epoch = 0
            reward_per_epoch = 0
            while epoch_steps < 10000:
                state = self.env.reset()
                done = False
                epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*i)
                steps = 0.
                ep_reward = 0.
                trace_list = []
                while not done:
                    value_t = []
                    # if not np.isscalar(state):
                        # state = tuple(state)
                    # epsilon greedy
                    if rng.rand() < epsilon:
                        maximum_action = rng.randint(0, action_size)
                    else:
                        vp = torch.Tensor(0,0)
                        va = torch.Tensor(0,0)
                        for action in range(action_size):
                            s_in = torch.Tensor(state)
                            a_in = torch.Tensor(action)
                            pred = self.net(s_in, a_in)
                            actual = knn(qec_table,(state,action))

                            value_t.append(pred.detach().numpy())
                            if sum(value_t)==0:
                                maximum_action = rng.randint(0, action_size)
                            else:
                                maximum_action = np.argmax(value_t)

                            vp = torch.cat((vp,pred),0)
                            va = torch.cat((va, torch.Tensor(actual)),0)
                        self.optimizer.zero_grad()
                        loss = self.loss(vp,va)
                        loss.backward()
                        self.optimizer.step()
                    next_state, reward, done , _ = self.env.step(maximum_action)

                    trace_list.append((state, maximum_action, reward, done))
                    state = next_state
                    ep_reward += reward # total reward for this episode: 1 if convergence
                    steps += 1.0
                reward_per_epoch += ep_reward
                epoch_steps += steps
                episodes_per_epoch += 1

                q_return = 0.
                for j in range(len(trace_list)-1, -1, -1):
                    node = trace_list[j]
                    q_return = q_return * ec_discount + node[2]
                    qec_table = update_table(qec_table,q_return,(node[0],node[1]))

                # train neural network

            self.total_reward.append(reward_per_epoch/episodes_per_epoch)
            self.total_sum_reward += reward_per_epoch
            # print('Average Reward: '+ str(sum(ep_avg_reward)/len(ep_avg_reward)))
            print('Average Epoch ' + str(i) + ' Reward: ' + str(total_reward[-1]))
            print('Total Reward: ' + str(total_sum_reward))

buffer_size = 100000
ec_discount = .99
min_epsilon = 0.01
decay_rate = 10000
epochs = 5000
continuous = True
knn = 11
environment = gym.make('MountainCar-v0')
rng = np.random.RandomState(123456)

EC = Episodic_Control(environment, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn)
EC.train()

# plot EC.total_reward for average reward over episodes
