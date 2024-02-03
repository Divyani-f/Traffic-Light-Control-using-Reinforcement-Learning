# Traffic Light Control using Reinforcement Learning

The Deep Reinforcement Learning Traffic Light Control project aims to provide an intelligent control system for regulating traffic flow at intersections. The system's goal is to minimize delays and lessen congestion.

## Code Structure

- **Main:** Contains the primary Python file responsible for the core functionality of the project.
  
- **Test:** Includes two test files to assess the model's performance.
  
- **Map:** Holds all the maps used for training the model.

## Overview

I trained a reinforcement learning model using Deep Q Learning with the TensorFlow library for the neural network component. The model's performance was evaluated using a traffic simulation, measuring average wait time and queue length before and after training.

## Installation

1. Clone the repository: `git clone https://github.com/Divyani-f/Traffic-Light-Control-using-Reinforcement-Learning.git`
2. Navigate to the project directory: `cd Traffic-Light-Control-using-Reinforcement-Learning`
3. Install dependencies: `pip install -r requirements.txt`
4. Sumo software Installation: Follow the steps from the [Sumo Documentation](https://sumo.dlr.de/docs/Installing/index.html)

## Usage

To run the project and observe the model's performance, follow these steps:

1. Navigate to the `Main` folder.
2. Run the main Python file to train the model using Deep Q Learning.
3. After training, observe the model's performance by running the traffic simulation.
4. Review the results, including average wait time and queue length. A detailed report is available in the `report` folder.

## Folder Structure

- `Main/`: Contains the main Python file for training the Deep Q Learning model.
- `Test/`: Includes two test files for assessing the model's performance.
- `Map/`: Holds all the maps used during the training process.

## Report

For a detailed report on the project, refer to the `report` folder.
