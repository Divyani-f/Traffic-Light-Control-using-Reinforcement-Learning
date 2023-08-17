import gym
from gym import spaces
import traci
import sumolib
import numpy as np
import xml.etree.ElementTree as ET
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        print("State size ---",state_size)

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_dim=3, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state,done))

    def act(self, state,epsilon): 
        if np.random.rand() <epsilon:
            print("hiii")
            return np.random.randint(self.action_size)
        else:
            print("not here")
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)
class TrafficSignalControlEnv(gym.Env):
    def parse_XML(self):
        tree = ET.parse('C:/Users/HP/Documents/RL-Project/Maps/map3/test.net.xml')
        root = tree.getroot()
        traffic_dict={}
        for tl in root.iter('tlLogic'):
            phases=[]
            for phase in tl.findall('phase'):
                state = phase.get('state')
                duration = phase.get('duration')
                phases.append(state)
            traffic_dict[tl.attrib.__getitem__('id')] =phases
                 
            
        return traffic_dict
    def __init__(self, config_file):
        #  SUMO configuration file
        self.Path="C:/Users/HP/Documents/RL-Project/Maps/map3/test.net.xml"
        self.SUMO_BINARY = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
        self.sumoCmd = [self.SUMO_BINARY, "-c", SUMO_CONFIG_FILE]
        self.sumo_config = sumolib.checkBinary('sumo')
        self.sumo_cmd = [self.sumo_config, '-c', config_file, '--time-to-teleport', '300', '--time-to-teleport.highway', '1200']
        self.traffic_dict= self.parse_XML()
        #  action space
        low = np.array([5, 0])
        high = np.array([50, 3])
        shape = (2,)
        #self.action_space = gym.spaces.Box(low=low, high=high, shape=shape)
        self.action_space =np.array([[5,0], [5,1], [5,2],[5,3],
                         [10,0], [10,1], [10,2],[10,3],
                         [20,0], [20,1], [20,2],[20,3],
                         [30,0], [30,1], [30,2],[30,3],
                         [40,0], [40,1], [40,2],[40,3],
                         [40,0], [40,1], [40,2],[40,3],])
        #Observation space
        self.observation_space=spaces.Dict({
        'wait_time':spaces.Discrete(8),
         'queue_length': spaces.Discrete(8)
        })
        wait_categories =[5000,10000,20000,30000,40000, 50000,70000,100000]
        len_categories= [100,500,1000,1500,2000,3000,5000,10000]
        self.observation_space.spaces['wait_time'].values = wait_categories
        self.observation_space.spaces['queue_length'].values = len_categories
    def reset(self):
        #traci.start(self.sumoCmd)
        
        pass

    def step(self, action):
        pass
        
    def render(self):
        pass
if __name__ == "__main__":
    #e=gym.make('TrafficSignalControlEnv-v0')
    print("here")
    SUMO_CONFIG_FILE = "C:/Users/HP/Documents/RL-Project/Maps/map3/map.sumo.cfg"
    e=TrafficSignalControlEnv(SUMO_CONFIG_FILE)
    #model = DQN(MlpPolicy, e, verbose=1)
    #Q - Learning
    state_size = e.observation_space['wait_time'].n * e.observation_space['queue_length'].n
    action_size = e.action_space.shape[0]
    print(action_size)
    print(state_size,action_size)
    agent = DQNAgent(state_size, action_size)
    delay=0
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.999
    exploration_decay_rate = 0.01 
    q_table =np.zeros((16,4))
    c=0
    print(e.action_space.shape)
    n_states = e.observation_space['wait_time'].n * e.observation_space['queue_length'].n
    
    n_actions = e.action_space.shape[0]
    #q_table = np.zeros((n_states, n_actions))
    q_table = np.zeros((8, 8, 20))
    state = e.reset()
    done = False
    step = 0
    prev_action=[40,3]
    prev_state={'wait_time':300,'queue_length':20}
    state= [300,20,1]
    state=np.reshape(state, [1, 3])
    sumoCmd = [e.SUMO_BINARY, "-c", SUMO_CONFIG_FILE]
    
    traffic_lights = list(e.traffic_dict.keys())

    for tl_id in traffic_lights:
        traci.start(sumoCmd)
        while step < 500:
            # Choose an action based on epsilon-greedy policy
            queue_length={}
            waiting_time={} 
            # if np.random.uniform() < epsilon:
            #     action=e.action_space[random.randint(0,19)]
            # else:
            #     print("here")
            #     action = np.argmax(prev_state[state[0], prev_state[1]])      
            
            action = agent.act(state,epsilon)
            phase=e.action_space[action][1]
            duration=e.action_space[action][0]
            traci.trafficlight.setPhase(tl_id, phase)
            traci.trafficlight.setPhaseDuration(tl_id, duration)
            traci.simulationStep()   
            tl_id = "gneJ11"
            #lane_id="-596401821#0"
            all_edge_ids = traci.edge.getIDList()
            lane_ids=traci.trafficlight.getControlledLanes(tl_id)

            for lane_id in lane_ids:
                if lane_id  not in queue_length.keys():
                    queue_length[lane_id]=traci.lane.getLastStepHaltingNumber(lane_id)
                else:
                    queue_length[lane_id] += traci.lane.getLastStepHaltingNumber(lane_id)             
                if lane_id not in waiting_time.keys():
                    waiting_time[lane_id]=traci.lane.getWaitingTime(lane_id)
                else:
                    waiting_time[lane_id] += traci.lane.getWaitingTime(lane_id)

            steps = traci.simulation.getTime()

            if steps>2000:
                traci.close()  
            
            wait_categories =[20,40,60,100,150, 200,300,300]
            len_categories= [2,5,7,9,10,15,17,20]
            queue_len= min(len_categories, key=lambda x: abs(x - sum(queue_length.values())))
            waiti_time=min(wait_categories, key=lambda x: abs(x - sum(waiting_time.values())))
            
            current_state= {'wait_time': waiti_time, 'queue_length': queue_len}
            next_state=[sum(waiting_time.values()),sum(queue_length.values()),traffic_lights.index(tl_id)]
            next_state=np.reshape(state, [1, 3])
            #print(next_state)
            reward=prev_state['wait_time']-current_state['wait_time']
            #next_state = np.reshape(next_state, [1, state_size])
            agent.remember(np.array(state), action, reward, np.array(next_state), done)
            
            # Take the chosen action and observe the next state and reward
            #next_state, reward, done, info = e.step(action)
            
            # Update Q-value using Q-learning equation
            index0=wait_categories.index(waiti_time)
            index1=len_categories.index(queue_len)
            prev_index0=wait_categories.index(prev_state['wait_time'])
            prev_index1=len_categories.index(prev_state['queue_length'])
            #actionIndex=np.where(e.action_space==action)
            #q_table[index0, index1,actionIndex ] += alpha * (reward + gamma * np.max(q_table[index0, index1]) - q_table[prev_index0, prev_index1, actionIndex])
            #print(sum(queue_length.values()),sum(waiting_time.values()))
                # Update state and step count
            prev_state = current_state
            state=next_state
            prev_action=action
            step += 1
            epsilon = epsilon_min + \
            (epsilon_decay - epsilon_min) * np.exp(-exploration_decay_rate * step)
        traci.close()
    if len(agent.memory) > 50:
        agent.replay(50)
    agent.save("model5.h5")
         
    traci.start(sumoCmd)
    step=0
    tl_id="gneJ2"
    while step < 500:
        traci.simulationStep()  
        action = agent.act(state,epsilon=0)
        phase=e.action_space[action][1]
        duration=e.action_space[action][0]
        print(phase, duration)
        traci.trafficlight.setPhase(tl_id, phase)
        traci.trafficlight.setPhaseDuration(tl_id, duration)
        for lane_id in lane_ids:
            if lane_id  not in queue_length.keys():
                queue_length[lane_id]=traci.lane.getLastStepHaltingNumber(lane_id)
            else:
                queue_length[lane_id] += traci.lane.getLastStepHaltingNumber(lane_id)             
            if lane_id not in waiting_time.keys():
                waiting_time[lane_id]=traci.lane.getWaitingTime(lane_id)
            else:
                waiting_time[lane_id] += traci.lane.getWaitingTime(lane_id)
        state=[sum(waiting_time.values()),sum(queue_length.values()),traffic_lights.index(tl_id)]
        print(state)
        state=np.reshape(state, [1, 3])
        step+=1
    print(waiting_time)
    print (queue_length)
    print("Total Waiting time --",sum(waiting_time.values()))
    print("Total Queue length --",sum(queue_length.values()))
    