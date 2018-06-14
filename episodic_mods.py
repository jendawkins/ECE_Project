import gym
import numpy as np
# import matplotlib.pyplot as plt
# __author__ = 'sudeep raja'
import numpy as np
# import _pickle as cPickle
import pickle
import heapq
from sklearn.neighbors import BallTree,KDTree
from neural_net import *
import time
from scipy.spatial.distance import cdist
import operator
#from sklearn.model_selection import GridSearchCV

class Episodic_Control():
    def __init__(self, environment, epochs, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn,lrr,filter,save_name):
        self.save_name = save_name
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
        #self.lr = lrr
        self.lr = lrr * (0.1 ** (self.epochs // 30))
        self.filter = filter
        # state_size = env.observation_space.shape[0]
        if continuous:
            self.state_dimension = self.env.observation_space.shape[0]
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_dimension = 1
            self.state_size = self.env.observation_space.n
        self.net = neural_net(self.state_dimension,self.action_size,1)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr = self.lr)
        self.counter = {}

    def knn_func(self,new):
        if len(self.qec_table)==0:
            return 0.0
        if new in self.qec_table.keys():
            return self.qec_table[new]
        states,actions = zip(*[key for key,item in self.qec_table.items() if key[1]==new[1]])
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
        dist, ind = tree.query(query_pt, k)
        value = 0
        for index in ind[0]:
            value += self.qec_table[(states[index],actions[index])]

        return value / knn

    def update_table(self,R,new):
        if new in self.qec_table.keys():
            if R>self.qec_table[new]:
                self.qec_table[new] = R
                self.counter[new] += 1
        else:
            self.qec_table[new] = R
            self.counter[new] = 1
        if self.filter:
            sa, goals = zip(*self.qec_table.items())
            states, actions = zip(*sa)
            # states,actions = zip(*[key for key,item in self.qec_table.items()])
            # goals = [item for key, item in self.qec_table.items()]

            state_std = np.std(np.array(states), axis = 0)
            state_rms = np.sqrt(np.sum(np.power(state_std,2)))

            idxs = np.where(cdist(np.array(states),np.array([new[0]]))<state_rms)[0]
            idxs2 = [idx for idx in idxs if np.array(actions)[idx] == new[1]]

            delta = np.std(np.array(goals)[idxs2])
            delt = np.sqrt(np.sum(np.power(delta,2)))
            const = delt

            # med_arr = np.median(np.array(goals)[idxs])
            med_arr = np.median(np.array(goals)[idxs2]) #changed!!

            for idx in idxs2:
                if goals[idx] + const < med_arr:
                    self.qec_table.pop((states[idx],actions[idx]))
                    self.counter.pop((states[idx],actions[idx]))
                    # import pdb; pdb.set_trace()
        if len(self.qec_table)>self.buffer_size:
            num_big = abs(self.buffer_size-len(self.qec_table))
            sorted_x = sorted(self.counter.items(), key=operator.itemgetter(1))
            if len(set(self.counter.values()))==1:
                id = np.random.choice(range(len(self.qec_table)),num_big)
                for idd in id:
                    del self.qec_table[(states[idd],actions[idd])]
                    del self.counter[(states[idd],actions[idd])]
            else:
                for keyy in sorted_x[:num_big]:
                    del self.qec_table[keyy[0]]
                    del self.counter[keyy[0]]

            # elif R + c > goals[idxs].all():

    def train_net(self,VISUALIZE):
        ep_avg_reward = []
        self.total_reward = []
        self.total_sum_reward = 0
        self.reward_per_ep = []
        for i in range(self.epochs):
            epoch_steps = 0
            episodes_per_epoch = 0
            reward_per_epoch = 0
            animate_this_episode = VISUALIZE and steps%10000==0
            start = time.time()
            while epoch_steps < 10000:
                if animate_this_episode:
                    self.env.render()
                    time.sleep(0.05)
                # if i < 4:
                # self.env.reset()
                # state = self.env.observation_space.sample()
                # self.env.env.state = state
                # else:
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
                    else:
                        state_t = state
                    if self.rng.rand() < epsilon:
                        maximum_action = self.rng.randint(0, self.action_size)
                    else:

                        for action in range(self.action_size):
                            if np.isscalar(state):
                                state = [state]
                            s_in = Variable(torch.Tensor([state]))
                            a_in = Variable(torch.Tensor([[action]]))
                            self.net.eval()
                            pred = self.net(s_in, a_in)

                            value_t.append(pred.data.numpy()[0][0])
                        if len(set(value_t))==1:
                            maximum_action = self.rng.randint(0, self.action_size)
                        else:
                            maximum_action = np.argmax(value_t)
                        # if i==2:
                            # import pdb; pdb.set_trace()
                    next_state, reward, done , _ = self.env.step(maximum_action)

                    trace_list.append((state_t, maximum_action, reward, done))
                    state = next_state
                    ep_reward += reward # total reward for this episode: 1 if convergence
                    steps += 1.0
                reward_per_epoch += ep_reward
                epoch_steps += steps
                episodes_per_epoch += 1
                self.reward_per_ep.append(ep_reward)
                q_return = 0.
                state_tensor = []
                action_tensor = []
                va = torch.Tensor(0,0)
                self.net.train()
                # update qec table
                stsss = [keyy[0] for keyy in self.qec_table.keys()]
                dd = np.std(np.array(stsss),axis = 0)
                dd2 =  np.sqrt(np.sum(np.power(dd,2)))
                const = dd2*np.exp(-i)
                for j in range(len(trace_list)-1, -1, -1):
                    node = trace_list[j]
                    q_return = q_return * self.ec_discount + node[2]

                    self.update_table(q_return,(node[0],node[1]))

                # train network on updated table
                s_a, va = zip(*[(key,item) for key,item in self.qec_table.items()])
                va = Variable(torch.Tensor(np.array(va)).unsqueeze(1))
                state_tensor, action_tensor = zip(*s_a)
                state_tensor = Variable(torch.Tensor(np.array(state_tensor)))
                if len(state_tensor.size())==1:
                    state_tensor = state_tensor.unsqueeze(1)
                if len(self.qec_table)==1:
                    self.net.eval()
                action_tensor = Variable(torch.Tensor(np.array(action_tensor)).unsqueeze(1))
                preds = self.net(state_tensor, action_tensor)
                self.optimizer.zero_grad()
                loss = self.loss(preds,va)
                loss.backward()
                self.optimizer.step()
            end = time.time()
            print(end - start)
            self.total_reward.append(reward_per_epoch/episodes_per_epoch)
            self.total_sum_reward += reward_per_epoch
            print('Average Epoch ' + str(i) + ' Reward: ' + str(self.total_reward[-1]))
            print('Total Reward: ' + str(self.total_sum_reward))
            print(len(self.qec_table))
            #with open(self.save_name + '.csv','a') as f:
                #f.write(str(self.total_reward[-1]) + ', ')

            #with open(self.save_name + '2.csv','a') as f:
                #f.write(str(self.reward_per_ep[-episodes_per_epoch:]) + ', ')
            #pickl_file = open(self.save_name + '.pkl','wb')
            #pickle.dump(self.qec_table,pickl_file)
            #pickl_file.close()
            #torch.save(self.net, self.save_name + '.pt')
            # print(len(self.qec_table))
            # print(self.qec_table)

buffer_size = 10000
ec_discount = .9
min_epsilon = 0.01
decay_rate = 1
epochs = 60
knn = 11
filter = True
learning_rate = 1e-2
rng = np.random.RandomState(123456)
environment = gym.make('LunarLander-v2')
VISUALIZE = False
save_name = 'LunarLander'

if VISUALIZE:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    environment = gym.wrappers.Monitor(environment, logdir, force=True, video_callable=lambda episode_id: episode_id%logging_interval==0)

rng = np.random.RandomState(123456)
continuous = isinstance(environment.observation_space, gym.spaces.Discrete)==False
# net = neural_net()
#(self, net, environment, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn)
EC = Episodic_Control(environment, epochs, rng, continuous, buffer_size, ec_discount, min_epsilon,
                decay_rate,knn,learning_rate, filter,save_name)
EC.train_net(VISUALIZE)

# plot EC.total_reward for average reward over episodes
