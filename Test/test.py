import os
import sys
import traci

# Path to your SUMO config file
SUMO_CONFIG_FILE = "C:/Users/HP/Documents/RL-Project/Maps/map2/map.sumo.cfg"

# Path to your SUMO binary
SUMO_BINARY = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"

# Start the SUMO simulation
sumoCmd = [SUMO_BINARY, "-c", SUMO_CONFIG_FILE]
traci.start(sumoCmd)

# Run the simulation
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    for veh_id in traci.vehicle.getIDList():
        waiting_time = traci.vehicle.getWaitingTime(veh_id)
        print(f"Vehicle {veh_id} waiting time: {waiting_time}")
        tl_id = "176086437"
        lane_id="-596401821#0"
        #print(type(traci.lanearea))
        all_edge_ids = traci.edge.getIDList()
        #print("All edge IDs:", all_edge_ids)

        #lane_id=traci.trafficlight.getControlledLanes(tl_id)[0]
        queue_length = traci.edge.getLastStepHaltingNumber(lane_id)
        print("Queue length at traffic light", tl_id, "is", queue_length)
# Stop the simulation and close the TraCI connection
traci.close()