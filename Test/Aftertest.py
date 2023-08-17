import os
import sys
import traci
import xml.etree.ElementTree as ET
import numpy as np 
from tensorflow.keras.models import load_model
def parse_XML():
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
#Load the model 
model1 = load_model('C:/Users/HP/Documents/RL-Project/model4.h5')

SUMO_CONFIG_FILE = "C:/Users/HP/Documents/RL-Project/Maps/map3/map.sumo.cfg"
step=0
# Path to your SUMO binary
SUMO_BINARY = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
sumoCmd = [SUMO_BINARY, "-c", SUMO_CONFIG_FILE]
traci.start(sumoCmd)
def act(state):
    q_values = model1.predict(state)
    return np.argmax(q_values[0])

action_space =np.array([[5,0], [5,1], [5,2],[5,3],
                         [10,0], [10,1], [10,2],[10,3],
                         [20,0], [20,1], [20,2],[20,3],
                         [30,0], [30,1], [30,2],[30,3],
                         [40,0], [40,1], [40,2],[40,3],
                         [50,0], [50,1], [50,2],[50,3],])
queue_length={}
waiting_time={}   
tl_id = "gneJ11"
state= [300,5,2]
state=np.reshape(state, [1, 3])
    #lane_id="-596401821#0"
all_edge_ids = traci.edge.getIDList()
lane_ids=list(traci.trafficlight.getControlledLanes(tl_id))   
traffic_dict=parse_XML()
traffic_lights = list(traffic_dict.keys())
steps=0         
while steps <500:
    traci.simulationStep()   
    #print(f"Vehicle {veh_id} waiting time: {waiting_time}")
    action = act(state)
    phase=action_space[action][1]
    duration=action_space[action][0]
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
        state=[sum(waiting_time.values()),sum(queue_length.values()),0]
        state=np.reshape(state, [1, 3])
        #steps = traci.simulation.getTime()

    steps=steps+1


traci.close()

print(waiting_time)
print (queue_length)
print("Total Waiting time --",sum(waiting_time.values()))
print("Total Queue length --",sum(queue_length.values()))