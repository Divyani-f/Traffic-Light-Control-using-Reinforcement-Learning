import gym
from gym import spaces
import traci
import sumolib

class TrafficSignalControlEnv(gym.Env):
    def __init__(self, config_file):
        # Load SUMO configuration file
        self.sumo_config = sumolib.checkBinary('sumo')
        self.sumo_cmd = [self.sumo_config, '-c', config_file, '--time-to-teleport', '300', '--time-to-teleport.highway', '1200']
        
        # Define action space and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            'waiting_time': spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
            'flow_rate': spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32),
            'time_since_last_change': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'time_since_last_vehicle': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'weather_conditions': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        })

    def reset(self):
        # Start SUMO simulation and connect to TraCI
        traci.start(self.sumo_cmd)
        self.step_count = 0

        # Initialize state
        self.state = {
            'waiting_time': np.zeros(4),
            'flow_rate': np.zeros(4),
            'time_since_last_change': np.zeros(1),
            'time_since_last_vehicle': np.zeros(1),
            'weather_conditions': np.zeros(3)
        }

        return self.state

    def step(self, action):
        # Set traffic signal phase
        traci.trafficlight.setPhase("TL", action)

        # Advance simulation by 1 step
        traci.simulationStep()
        self.step_count += 1

        # Update state
        self.state['waiting_time'] = np.array([
            traci.edge.getWaitingTime('NtoS'),
            traci.edge.getWaitingTime('StoW'),
            traci.edge.getWaitingTime('WtoE'),
            traci.edge.getWaitingTime('EtoN')
        ])
        self.state['flow_rate'] = np.array([
            traci.edge.getLastStepVehicleNumber('NtoS'),
            traci.edge.getLastStepVehicleNumber('StoW'),
            traci.edge.getLastStepVehicleNumber('WtoE'),
            traci.edge.getLastStepVehicleNumber('EtoN')
        ])
        self.state['time_since_last_change'] = np.array([
            traci.trafficlight.getTimeSinceLastSwitch('TL')
        ])
        self.state['time_since_last_vehicle'] = np.array([
            traci.simulation.getTime() - traci.edge.getLastStepVehicleTime('NtoS')
        ])
        self.state['weather_conditions'] = np.array([
            # Get weather data from an external source, such as a weather API
            # For demonstration purposes, we just use random noise here
            np.random.normal(0, 1),
            np.random.normal(0, 1),
            np.random.normal(0, 1)
        ])

        # Calculate reward based on wait time
        total_wait_time = np.sum(self.state['waiting_time'])
        reward = -total_wait_time

        # Check if episode is done
        done = self.step_count >= 1000

        return self.state, reward, done
