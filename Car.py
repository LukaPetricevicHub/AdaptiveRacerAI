import numpy as np
import random
import cv2
from keras import callbacks
from keras.layers import Convolution2D, Dense, Flatten, Input
from keras.models import Model
from keras import backend as K

# Agent 
max_reward = 10
action_repeat_num = 8
max_num_episodes = 1000
max_num_steps = action_repeat_num * 100
gamma = 0.99
max_eps = 0.1
min_eps = 0.02
EXPLORATION_STOP = int(max_num_steps * 10)
_lambda_ = -np.log(0.001) / EXPLORATION_STOP
UPDATE_TARGET_FREQUENCY = int(50)
batch_size = 128

class Agent:
    steps = 0
    epsilon = max_eps
    memory = []

    def __init__(self, num_states, num_actions, img_dim, model_path):
        self.num_states = num_states
        self.num_actions = num_actions
        self.DQN = DQN(num_states, num_actions, model_path)
        self.no_state = np.zeros((84, 96, 1))
        self.x = np.zeros((batch_size,) + img_dim)
        self.y = np.zeros([batch_size, num_actions])
        self.errors = np.zeros(batch_size)
        self.rand = False

        self.agent_type = 'Learning'
        self.maxEpsilon = max_eps

    def act(self, s):
        print("Shape of state before prediction:", s.shape)
        if random.random() < self.epsilon:
            random_action = np.array([np.random.uniform(-1, 1), 1, 0], dtype=np.float32)  
            self.rand = True
            return random_action
        else:
            act_soft = self.DQN.predict_single_state(s)
            best_act = np.argmax(act_soft)

            action = np.array([float(best_act), 1.0, 0.0], dtype=np.float32)  
            action_soft = act_soft

        self.rand = False
        return action, action_soft

    def compute_targets(self, batch):
        states = np.array([rec[1][0] for rec in batch])
        states_ = np.array([(self.no_state if rec[1][3] is None else rec[1][3]) for rec in batch])

        states = states.reshape((-1, 84, 96, 1))
        states_ = states_.reshape((-1, 84, 96, 1))

        print("Shape of states:", states.shape)
        print("Shape of states_:", states_.shape)

        p = self.DQN.predict(states)
        p_ = self.DQN.predict(states_, target=False)
        p_t = self.DQN.predict(states_, target=True)
        act_ctr = np.zeros(self.num_actions)

        for i in range(len(batch)):
            rec = batch[i][1]
            s = rec[0]
            a = np.array(rec[1], dtype=np.float32)  
            r = rec[2]
            s_ = rec[3]

            t = p[i]
            a = int(a[0])  
            act_ctr[a] += 1

            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + gamma * p_t[i][np.argmax(p_[i])]  

            self.x[i] = s
            self.y[i] = t

            if self.steps % 20 == 0 and i == len(batch) - 1:
                print('t', t[a], 'r: %.4f' % r, 'mean t', np.mean(t))
                print('act ctr: ', act_ctr)

            self.errors[i] = abs(oldVal - t[a])

        return (self.x, self.y, self.errors)
    

    

    def observe(self, sample):  
        
        if sample[3] is not None:
            sample = (sample[0], sample[1], sample[2], preprocess_state(sample[3]))
        else:
            sample = (sample[0], sample[1], sample[2], self.no_state)

        _, _, errors = self.compute_targets([(0, sample)])
        self.memory.append(errors[0])
 
        self.memory[-1][1] = np.array(self.memory[-1][1], dtype=np.float32)


    def replay(self):
        if len(self.memory) >= batch_size:
            indices = random.sample(range(len(self.memory)), batch_size)
            batch = [self.memory[i] for i in indices]
            x, y, errors = self.compute_targets(batch)
            for i in range(len(indices)):
                self.memory[indices[i]] = errors[i]

            self.DQN.train(x, y)

class RandomAgent:
    memory = []
    exp = 0
    steps = 0

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.agent_type = 'Learning'
        self.rand = True

    def act(self, s):
        best_act = np.random.randint(self.num_actions)
        return best_act, best_act

    def observe(self, sample):  
        _, _, errors, _ = self.compute_targets([(0, sample)])
        self.memory.append(errors[0])

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.DQN.target_model_update()
        self.steps += 1
        self.epsilon = min_eps + (self.maxEpsilon - min_eps) * np.exp(-1 * _lambda_ * self.steps)

    def replay(self):
        pass

class DQN:
    def __init__(self, num_states, num_actions, model_path):
        self.num_states = num_states
        self.num_actions = num_actions
        self.model = self.build_model()  
        self.model_ = self.build_model()  

        self.model_chkpoint_1 = model_path + "CarRacing_DDQN_model_1.h5"
        self.model_chkpoint_2 = model_path + "CarRacing_DDQN_model_2.h5"

        self.callbacks_list = self.build_callbacks()

    def build_callbacks(self):
        save_best = callbacks.ModelCheckpoint(self.model_chkpoint_1, monitor='loss', verbose=1, save_best_only=True, save_freq=20)
        save_per = callbacks.ModelCheckpoint(self.model_chkpoint_2, monitor='loss', verbose=1, save_best_only=False, save_freq=400)

        return [save_best, save_per]

    def build_model(self):
        states_in = Input(shape=self.num_states, name='states_in')  
        x = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu')(states_in)
        x = Convolution2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
        x = Convolution2D(64, (3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self.num_actions, activation='linear')(x)

        model = Model(inputs=states_in, outputs=predictions)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def target_model_update(self):
        self.model_.set_weights(self.model.get_weights())

    def predict(self, states, target=False):
        model = self.model_ if target else self.model

        if len(states.shape) == 3:
            states = states.reshape((1,) + states.shape)  

        return model.predict(states)

    def predict_single_state(self, s, target=False):
        model = self.model_ if target else self.model

        return model.predict(s)[0]

    def train(self, x, y):
        self.model.fit(x, y, epochs=1, verbose=0, callbacks=self.callbacks_list)

   

def run_experiment(agent, environment):
    for episode in range(max_num_episodes):
        state = environment.reset()
        total_reward = 0
        episode_done = False
        step = 0

        while not episode_done:
            
            state = preprocess_state(np.array(state))  

            action_result = agent.act(state)

            if agent.rand:
                action = action_result
            else:
                action, _ = action_result

            next_state, reward, episode_done, *_ = environment.step(action)

            sample = (state, action, reward, None if episode_done else next_state)
            agent.observe(sample)

            state = next_state
            total_reward += reward

            if step >= max_num_steps:
                episode_done = True

            step += 1

        agent.replay()

        print("Episode {}: Total Reward: {}".format(episode + 1, total_reward))

    environment.close()


def preprocess_state(state):
    state_data = state

    state_data = state_data[:84, :]  
    state_data = state_data / 255.0  

    state_data = cv2.cvtColor(state_data, cv2.COLOR_BGR2GRAY)

    state_data = cv2.resize(state_data, (96, 96))  

    state_data = state_data.reshape((96, 96, 1))  

    return state_data



if __name__ == "__main__":

    num_states = (84, 96, 1)
    num_actions = 3
    img_dim = (84, 96, 1)
    model_path = './'  

    agent = Agent(num_states, num_actions, img_dim, model_path)

    import gym
    environment = gym.make('CarRacing-v2')

    run_experiment(agent, environment) 