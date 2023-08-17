import os
import sys
import traci
import xml.etree.ElementTree as ET

# Path to your SUMO config file
SUMO_CONFIG_FILE = "C:/Users/HP/Documents/RL-Project/map3/map.sumo.cfg"
step=0
# Path to your SUMO binary
SUMO_BINARY = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"

# Start the SUMO simulation
sumoCmd = [SUMO_BINARY, "-c", SUMO_CONFIG_FILE]
traci.start(sumoCmd)
tl_id = "gneJ2"
# Run the simulation
tree = ET.parse('C:/Users/HP/Documents/RL-Project/Maps/map3/test.net.xml')
root = tree.getroot()
traffic_lights = traci.trafficlight.getIDList()
traffic_dict={}

def get_closest_category(values, categories):
    for key, value in values.items():
        closest_category = min(categories, key=lambda x: abs(x - value))
        values[key] = closest_category
    return values
for id in traffic_lights:
    for tl in root.iter('tlLogic'):
        phases=[]
        for phase in tl.findall('phase'):
            state = phase.get('state')
            duration = phase.get('duration')
            phases.append(state)
            traffic_dict[tl.attrib.__getitem__('id')]  =phases
queue_length={}
waiting_time={}   
tl_id = "gneJ2"
    #lane_id="-596401821#0"
all_edge_ids = traci.edge.getIDList()
lane_ids=list(traci.trafficlight.getControlledLanes(tl_id))   
steps=0         
while steps <500:
    traci.simulationStep()   
    #print(f"Vehicle {veh_id} waiting time: {waiting_time}")

    for lane_id in lane_ids:
        if lane_id  not in queue_length.keys():
            queue_length[lane_id]=traci.lane.getLastStepHaltingNumber(lane_id)
        else:
            queue_length[lane_id] += traci.lane.getLastStepHaltingNumber(lane_id)             
        if lane_id not in waiting_time.keys():
            waiting_time[lane_id]=traci.lane.getWaitingTime(lane_id)
        else:
            waiting_time[lane_id] += traci.lane.getWaitingTime(lane_id)

        #steps = traci.simulation.getTime()

    steps=steps+1


traci.close()
print(waiting_time)
print (queue_length)
print("Total Waiting time --",sum(waiting_time.values()))
print("Total Queue length --",sum(queue_length.values()))
    