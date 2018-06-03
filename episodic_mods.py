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
#import pdb; pdb.set_trace

class Episodic_Control():
    def __init__(self, environment, epochs, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn):

        self.env = environment
        self.rng = rng
        self.buffer_size = buffer_size
        self.ec_discount = ec_discount
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.knn = knn
        self.qec_table = {}
        self.filt_qec_table = {}
        self.action_size = self.env.action_space.n
        self.epochs = epochs

        # state_size = env.observation_space.shape[0]
        if continuous:
            self.state_dimension = self.env.observation_space.shape[0]
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_dimension = 1
            self.state_size = self.env.observation_space.n

        self.net = neural_net(self.state_dimension,self.action_size,1)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr = .01)

    def knn_func(self,new):
        # import pdb; pdb.set_trace()
        if len(self.qec_table)==0:
            return 0.0
        if new in self.qec_table.keys():
            return self.qec_table[new]
        states,actions = zip(*[key for key,item in self.qec_table.items()])
        if len(self.qec_table) < self.knn:
            k = len(self.qec_table)
        else:
            k = self.knn
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
            value += self.qec_table[(states[index],actions[index])]

        return value / knn

    def update_table(self,R,new):
        # import pdb; pdb.set_trace()
        if new in self.qec_table.keys():
            if R>self.qec_table[new]:
                self.qec_table[new] = R
        else:
            self.qec_table[new] = R

    def filter_ds(self):
        states,actions = zip(*[key for key,item in self.qec_table.items()])
        g = [item for key,item in self.qec_table.items()]
        states_j, actions_j = zip(*[key for key,item in self.filt_qec_table.items()])
        g_j = [item for key,item in self.filt_qec_table.items()]
        delta = 1
        c = 1 #change this to decrease in size with number of epochs
        for i in range(len(states)):
            for j in range(len(states_j)):
                if actions[i] == actions_j[j] and abs(states[i]-states_j[j]) < delta:
                    if g[i] + c < g_j[j]:
                        new = (states[i], actions[i])
                        self.filt_qec_table[new] = g[i]
                if g_j[j] < np.median(g_j): #medium
                    old = (states_j[j], actions_j[j])
                    self.filt_qec_table.pop(old, None)

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
                    if not np.isscalar(state):
                        state_t = tuple(state)
                    # epsilon greedy
                    if rng.rand() < epsilon:
                        maximum_action = rng.randint(0, self.action_size)
                    else:
                        vp = torch.Tensor(0,0)
                        va = torch.Tensor(0,0)
                        for action in range(self.action_size):
                            s_in = Variable(torch.Tensor([state]))
                            a_in = Variable(torch.Tensor([[action]]))
                            # import pdb; pdb.set_trace()
                            pred = self.net(s_in, a_in)
                            actual = self.knn_func((state_t,action))

                            value_t.append(pred.data.numpy())
                            if sum(value_t)==0:
                                maximum_action = rng.randint(0, self.action_size)
                            else:
                                maximum_action = np.argmax(value_t)

                            vp = torch.cat((vp,pred),0)
                            va = torch.cat((va, torch.Tensor([[actual]])),0)
                        self.optimizer.zero_grad()
                        loss = self.loss(Variable(vp),Variable(va))
                        loss.backward()
                        self.optimizer.step()
                    next_state, reward, done , _ = self.env.step(maximum_action)

                    trace_list.append((state_t, maximum_action, reward, done))
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
                    self.update_table(q_return,(node[0],node[1]))

                # train neural network here instead?

            self.total_reward.append(reward_per_epoch/episodes_per_epoch)
            self.total_sum_reward += reward_per_epoch
            # print('Average Reward: '+ str(sum(ep_avg_reward)/len(ep_avg_reward)))
            print('Average Epoch ' + str(i) + ' Reward: ' + str(self.total_reward[-1]))
            print('Total Reward: ' + str(self.total_sum_reward))

buffer_size = 100000
ec_discount = .99
min_epsilon = 0.01
decay_rate = 10000
epochs = 5000
continuous = True
knn = 11
environment = gym.make('MountainCar-v0')
rng = np.random.RandomState(123456)
# net = neural_net()
#(self, net, environment, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn)
EC = Episodic_Control(environment, epochs, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn)
EC.train()

# plot EC.total_reward for average reward over episodes