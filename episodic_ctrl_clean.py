import gym
import numpy as np
import matplotlib.pyplot as plt
# __author__ = 'sudeep raja'
import numpy as np
# import _pickle as cPickle
import pickle
import heapq
import scipy
from sklearn.neighbors import BallTree,KDTree
from scipy.spatial.distance import cdist
import cv2

class Episodic_Control():
    def __init__(self, environment, epochs, rng, continuous, buffer_size, ec_discount, min_epsilon, decay_rate,knn,images,filter_buffer,load_data):
        self.filter = filter_buffer
        self.counter = {}
        self.images = images
        self.env = environment
        self.rng = rng
        self.buffer_size = buffer_size
        self.ec_discount = ec_discount
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.knn = knn
        self.qec_table = {}
        self.action_size = self.env.action_space.n
        self.epochs = epochs
        # state_size = env.observation_space.shape[0]
        if continuous:
            self.state_dimension = self.env.observation_space.shape[0]
            self.state_size = self.env.observation_space.shape[0]
        else:
            self.state_dimension = 1
            self.state_size = self.env.observation_space.n
        if self.images:

            self.state_dimension = self.env.observation_space.shape
            self._initialize_projection_function(84*84)

        if load_data:
            file_table = open('MtCar_Net_Filter.pkl','rb')
            self.qec_table = pickle.load(file_table)
            file_table.close()

    def knn_func(self,new):
        if len(self.qec_table)==0:
            return 0.0
        if new in self.qec_table.keys():
            return self.qec_table[new]
        states,actions = zip(*[key for key,item in self.qec_table.items() if key[1]==new[1]])
        if len(states) < self.knn:
            k = len(states)
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

    def update_table(self,R,new,const):
        if new in self.qec_table.keys():
            if R>self.qec_table[new]:
                self.qec_table[new] = R
        # else:
        #     self.qec_table[new] = R
        elif len(self.qec_table)<10 or not self.filter:
            self.qec_table[new] = R
            self.counter[new] = 1
        else:
            states,actions = zip(*[key for key,item in self.qec_table.items()])
            goals = [item for key, item in self.qec_table.items()]
            delta = np.std(np.array(states),axis = 0)
            delt = np.sqrt(delta[0]**2 + delta[1]**2)
            idxs = np.where(cdist(np.array(states),np.array([new[0]]))<delt)[0]
            med_arr = np.median(np.array(goals)[idxs])
            if new[1] not in np.array(actions)[idxs] or R + const > med_arr:
                # if R + c < goals[idxs].all():
                    # self.qec_table[]
                self.qec_table[new] = R
                self.counter[new] = 1
            for idx in idxs:
                if goals[idx] + const < med_arr:
                    # import pdb; pdb.set_trace()
                    self.qec_table.pop((states[idx],actions[idx]))
                    self.counter.pop((states[idx],actions[idx]))
        if len(self.qec_table)>self.buffer_size:
            if len(set(self.counter.keys()))==1:
                id = np.random.randint(0,len(self.qec_table))
                del self.qec_table[(states[id],actions[id])]
                del self.counter[(states[id],actions[id])]
            else:
                del self.qec_table[min(self.counter, key=self.counter.get)]
                del self.counter[min(self.counter, key=self.counter.get)]


    def _initialize_projection_function(self, dimension_observation):
        self.matrix_projection = self.rng.randn(64,dimension_observation).astype(np.float32)

    def rgb2gray(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def train(self,VISUALIZE):
        ep_avg_reward = []
        self.reward_per_ep = []
        self.total_reward = []
        self.total_sum_reward = 0
        for i in range(self.epochs):
            epoch_steps = 0
            episodes_per_epoch = 0
            reward_per_epoch = 0
            while epoch_steps < 10000:
                # state = self.env.reset()
                self.env.reset()
                state = self.env.observation_space.sample()
                self.env.env.state = state
                    # state = np.reshape(state,(len(state),1))
                done = False
                epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*i)
                steps = 0.
                ep_reward = 0.
                trace_list = []
                animate_this_episode = VISUALIZE and steps%10000==0
                while not done:
                    if animate_this_episode:
                        self.env.render()
                        time.sleep(0.05)
                    if self.images:
                        state = self.rgb2gray(state)
                        state = scipy.misc.imresize(state, size=(84,84))
                        state = np.dot(self.matrix_projection, state.flatten())
                    value_t = []
                    if not np.isscalar(state):
                        state = tuple(state)
                    # epsilon greedy
                    if self.rng.rand() < epsilon:
                        maximum_action = self.rng.randint(0, self.action_size)
                    else:
                        for action in range(self.action_size):
                            value_t.append(self.knn_func((state,action)))
                        if len(set(value_t))==1:
                            # print('values_equal')
                            maximum_action = self.rng.randint(0, self.action_size)
                        else:
                            # print('values unequal')
                            # import pdb; pdb.set_trace()
                            maximum_action = np.argmax(value_t)
                        # import pdb; pdb.set_trace()
                    next_state, reward, done , _ = self.env.step(maximum_action)
                    trace_list.append((state, maximum_action, reward, done))
                    state = next_state
                    ep_reward += reward # total reward for this episode: 1 if convergence
                    steps += 1.0
                self.reward_per_ep.append(ep_reward)
                reward_per_epoch += ep_reward
                epoch_steps += steps
                episodes_per_epoch += 1
                # print(ep_reward)
                q_return = 0.
                for j in range(len(trace_list)-1, -1, -1):
                    node = trace_list[j]
                    q_return = q_return * self.ec_discount + node[2]
                    self.update_table(q_return,(node[0],node[1]),0)
            self.total_reward.append(reward_per_epoch/episodes_per_epoch)
            self.total_sum_reward += reward_per_epoch
            # print('Average Reward: '+ str(sum(ep_avg_reward)/len(ep_avg_reward)))
            print('Average Epoch ' + str(i) + ' Reward: ' + str(self.total_reward[-1]))
            print('Total Reward: ' + str(self.total_sum_reward))
            with open('mtcar.csv','a') as f:
                f.write(str(self.total_reward[-1]))
            with open('mtcar2.csv','a') as f:
                f.write(str(self.reward_per_ep[-episodes_per_epoch:]))
            if not VISUALIZE:
                pickl_file = open('mtcar2.pkl','wb')
                pickle.dump(self.qec_table,pickl_file)
                pickl_file.close()
            # with open('pacman.pkl','w') as pp:
                # pp.dump()

buffer_size = 100000
ec_discount = .99
min_epsilon = 0.01
decay_rate = 1000
epochs = 5000
# continuous = False
knn = 11
# environment = gym.make('MountainCar-v0')
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
# environment = gym.make('FrozenLakeNotSlippery-v0')
# environment = gym.make('CartPole-v0')
environment = gym.make('MountainCar-v0')
continuous = isinstance(environment.observation_space, gym.spaces.Discrete)==False
rng = np.random.RandomState(123456)

VISUALIZE = False

if VISUALIZE:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    environment = gym.wrappers.Monitor(environment, logdir, force=True, video_callable=lambda episode_id: episode_id%logging_interval==0)

images = False
filter_buffer = True
load_data = False
EC = Episodic_Control(environment, epochs, rng, continuous, buffer_size,
                ec_discount, min_epsilon, decay_rate,knn,images,filter_buffer,load_data)
EC.train(VISUALIZE)

# plot EC.total_reward for average reward over episodes
